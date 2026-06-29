"""NKI kernels for the VAE CausalConv3d temporal feature-cache.

The WAN VAE decodes video causally: each CausalConv3d with a temporal padding
keeps the last CACHE_T (=2) frames of its input so the next chunk's convolution
can prepend them instead of recomputing. In streaming use (decode a few frames
at a time) this cache is updated on EVERY chunk, so doing the shift/copy on-device
with NKI — instead of host-side torch.cat/slice that round-trips HBM — removes a
recurring copy from the hot path.

Two ops, mirroring rolling_forcing's design but adapted to THIS repo's
`torch_neuronx.nki_hop.wrap_nki` convention (pure @nki.jit, no nki_op/_compile):

  shift:  cache holds CACHE_T frames flattened as (C, CACHE_T*HW). When the new
          chunk has T < CACHE_T frames, slide the cache left by HW and write the
          new frame(s) into the tail. Keeps the most-recent CACHE_T frames.
  copy:   when the new chunk has >= CACHE_T frames, just copy its last CACHE_T
          frames into the cache (no shift needed).

Layout: cache (C, CACHE_T*HW) bf16; x (C, T*HW) bf16. HW = H*W (one frame).

SBUF is small on-chip SRAM — a whole frame (HW ~ 788K elems at full VAE res) does
NOT fit. So both kernels TILE the free (column) dimension into _TILE_F chunks as
well as the partition dimension into _TILE_P (=128) chunks; each staged DMA buffer
is at most (_TILE_P, _TILE_F). (v1 staged a full row and hit "Allocated memory out
of bound" on device.)
"""
import nki
import nki.language as nl
import nki.isa as nisa

_TILE_P = 128
_TILE_F = 8192  # column chunk that fits SBUF: 128 x 8192 bf16 = 2MB


def _copy_cols(dst, src, dst_off, src_off, ncols, p0, psz):
    """DMA src[p0:p0+psz, src_off:src_off+ncols] -> dst[..., dst_off:...], tiled."""
    n_f = (ncols + _TILE_F - 1) // _TILE_F
    for f_i in nl.affine_range(n_f):
        c0 = f_i * _TILE_F
        csz = min(_TILE_F, ncols - c0)
        buf = nl.ndarray((psz, csz), dtype=dst.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=buf[:, :], src=src[nl.ds(p0, psz), nl.ds(src_off + c0, csz)])
        nisa.dma_copy(dst=dst[nl.ds(p0, psz), nl.ds(dst_off + c0, csz)], src=buf[:, :])


@nki.jit
def causal_conv3d_cache_shift(cache, x, HW):
    """Shift cache left by one frame (HW cols) and append x's frame in the tail.

    Used when the incoming chunk has fewer than CACHE_T frames: the oldest frame
    is dropped, remaining frames slide left, the new frame lands at the end.

    Args:
        cache: (C, CACHE_T*HW) bf16 — current temporal cache
        x:     (C, HW)         bf16 — the new single frame to append
        HW:    int             — H*W (cols per frame)
    Returns:
        out:   (C, CACHE_T*HW) bf16 — shifted cache
    """
    C = cache.shape[0]
    cache_cols = cache.shape[1]
    out = nl.ndarray((C, cache_cols), dtype=cache.dtype, buffer=nl.shared_hbm)
    num_p = (C + _TILE_P - 1) // _TILE_P
    keep = cache_cols - HW  # cols carried over from the slide

    for p_i in nl.affine_range(num_p):
        p0 = p_i * _TILE_P
        psz = min(_TILE_P, C - p0)
        # slide: out[:, :keep] = cache[:, HW:]
        _copy_cols(out, cache, 0, HW, keep, p0, psz)
        # append: out[:, keep:] = x[:, :HW]
        _copy_cols(out, x, keep, 0, HW, p0, psz)
    return out


@nki.jit
def causal_conv3d_cache_copy(cache, x):
    """Copy x's last CACHE_T frames into the cache (chunk has >= CACHE_T frames).

    Args:
        cache: (C, CACHE_T*HW) bf16
        x:     (C, T*HW)       bf16 — full chunk; take its last cache_cols columns
    Returns:
        out:   (C, CACHE_T*HW) bf16
    """
    C = cache.shape[0]
    cache_cols = cache.shape[1]
    x_off = x.shape[1] - cache_cols
    out = nl.ndarray((C, cache_cols), dtype=cache.dtype, buffer=nl.shared_hbm)
    num_p = (C + _TILE_P - 1) // _TILE_P

    for p_i in nl.affine_range(num_p):
        p0 = p_i * _TILE_P
        psz = min(_TILE_P, C - p0)
        # out[:, :] = x[:, x_off:]
        _copy_cols(out, x, 0, x_off, cache_cols, p0, psz)
    return out
