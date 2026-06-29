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
Both kernels return the (mutated) cache as a fresh HBM tensor.
"""
import nki
import nki.language as nl
import nki.isa as nisa

_TILE_P = 128


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

    for p_i in nl.affine_range(num_p):
        p0 = p_i * _TILE_P
        psz = min(_TILE_P, C - p0)
        # slide: out[:, :cache_cols-HW] = cache[:, HW:]
        buf = nl.ndarray((psz, cache_cols - HW), dtype=cache.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=buf[:, :], src=cache[nl.ds(p0, psz), HW:])
        nisa.dma_copy(dst=out[nl.ds(p0, psz), 0:cache_cols - HW], src=buf[:, :])
        # append: out[:, cache_cols-HW:] = x
        xbuf = nl.ndarray((psz, HW), dtype=cache.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=xbuf[:, :], src=x[nl.ds(p0, psz), :])
        nisa.dma_copy(dst=out[nl.ds(p0, psz), cache_cols - HW:cache_cols], src=xbuf[:, :])
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
        buf = nl.ndarray((psz, cache_cols), dtype=cache.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=buf[:, :], src=x[nl.ds(p0, psz), x_off:])
        nisa.dma_copy(dst=out[nl.ds(p0, psz), :], src=buf[:, :])
    return out
