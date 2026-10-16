# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Constant-Max Attention Kernel

Computes the UNNORMALIZED partial attention

    [Σ P·V | Σ P·1]     where    P = exp(Q @ K.T * softmax_scale - c)

and folds it into a caller-owned SBUF accumulator. The two matmuls are referred to throughout as:
  - MM1: Q @ K.T (scores)
  - MM2: P @ [V | 1]  (weighted value accumulation, plus the sum_exp column)

`c` is the per-ROW softmax bound c_i = softmax_scale·‖q_i‖·max_j‖k_j‖, supplied by the caller in
``c_row_hbm`` in natural token order. Taking the max as given rather than deriving it online
eliminates:
  1. The DMA transpose between MM1 and MM2, which serializes with collective
     communication (CC) and blocks CC overlap.
  2. The online softmax max-reduction pass.

With no online max the scores stay K-stationary ([Sk-partition, Sq-free]), which is what lets a
ring driver's KV-rotation collective overlap this compute. The accumulator is SBUF-resident across
ring steps and the caller normalizes once at the end, so there is no HBM round-trip per step and no
separate cross-step reduction pass.
"""

from typing import Optional

import nki.isa as nisa
import nki.language as nl

from ...core.utils.kernel_assert import kernel_assert
from .. import neurotile as nt

# Tiling constants
_P = nl.tile_size.pmax  # 128: partition dimension
_SK_BLOCK = 2048  # KV streaming block size (16 tiles of 128)
_O_TILE_SIZE = nl.tile_size.gemm_stationary_fmax  # 128: output tile size along Sq (MM2 stationary)
_KV_TILES_PER_BLOCK = _SK_BLOCK // _P  # 16: number of K/V tiles per KV block

# PSUM bank allocation:
#   _MM1_BUFFER_BANKS (2) banks hold the MM1 score tiles, so the next KV tile's MM1 can start while
#   the current scores are still being read. The rest, at most _MAX_O_AUG_BANKS_CAP (8), hold the
#   O_aug accumulators: _max_o_aug_banks() = min(psum_num_banks - 2, 8).
#
# O_aug is the "augmented" output tile: 128 query rows by d+1 columns, [Σ P·V | Σ P·1]. MM2
# multiplies the probabilities P by [V | 1], so columns 0..d-1 accumulate the weighted output and
# column d accumulates the softmax denominator (sum_exp), which the final normalization divides by.
#
# Q and O blocks share the same Sq-dimension size. O tiles are staged in PSUM and
# accumulate across all KV blocks. The number of available O_aug banks is derived at
# trace time from nl.tile_size.psum_num_banks. Each O_aug tile takes its own bank and covers
# gemm_stationary_fmax (128) query rows, so the maximum O block is _max_o_aug_banks() × 128 Sq elements.
#
# On trn2 (gen3), MM1 produces f32 PSUM and the moving operand (Q) is limited to
# gemm_moving_fmax = 512, so we choose max Q/O block = 512. On trn3 (gen4), MM1
# produces bf16 PSUM with no such limit, so the full O_aug capacity is used.
#
# Q stays resident in SBUF while we stream all KV blocks, prefetching the next block
# behind the current block's compute.

# Trace-size budget for the statically unrolled partial attention, in emitted instructions summed
# over every head and ring step of one kernel. Every loop here is unrolled at trace time, so the
# emitted BIR grows as heads x ring steps x Q-blocks x KV tiles, and the serialized BIR body's size
# is a uint32 ("colz: body blob exceeds uint32 offset range" past 4 GiB). Measured on trn2
# (nki 0.6.0): ~950 bytes of body per instruction, so 1M instructions is ~22% of the limit. Wan2.2
# 1080p (10 heads, CP8, 21105 tokens/rank) emits ~4.4M — 98% — when fully unrolled; 720p emits
# ~0.9M and stays unrolled.
_MAX_STATIC_ATTN_INSTRUCTIONS = 1_000_000
# Instructions emitted per (Q-block, KV tile) pair, not counting the MM2s (one per 128-row output
# tile): V copy, MM1, subtract and exp. choose_q_group multiplies it out to estimate the fully
# unrolled kernel's instruction count, which decides between the unrolled and rolled paths and sets
# the rolled group size. It counts NKI calls, not hardware instructions; the ~950 bytes per
# instruction above was calibrated with it.
_INSTRUCTIONS_PER_KV_TILE = 4

# PSUM banks reserved for the MM1 score tiles, and the cap on O_aug banks (see the PSUM notes above).
_MM1_BUFFER_BANKS = 2
_MAX_O_AUG_BANKS_CAP = 8


def _max_o_aug_banks():
    """PSUM banks available for O_aug accumulators: 6 on trn2 and trn3 (8 banks - 2 for MM1)."""
    return min(nl.tile_size.psum_num_banks - _MM1_BUFFER_BANKS, _MAX_O_AUG_BANKS_CAP)


def _max_sq_block_size():
    """Q/O block size along Sq: 512 on trn2 (gen3, f32 PSUM caps the MM1 moving free size),
    _max_o_aug_banks() x 128 = 768 on trn3 (gen4, bf16 PSUM). See the PSUM bank notes above."""
    has_bf16_psum = nisa.get_nc_version() >= nisa.nc_version.gen4
    return _max_o_aug_banks() * _O_TILE_SIZE if has_bf16_psum else nl.tile_size.gemm_moving_fmax


def choose_q_group(num_heads, num_steps, sq, sk):
    """Q-blocks per rolled loop iteration, or None to keep the fully unrolled path.

    Estimate. For every head and ring step, one Q-block of max_sq rows emits, per KV tile,
    _INSTRUCTIONS_PER_KV_TILE instructions plus one MM2 per 128-row output tile. So one Q-block
    costs

        per_q_block = num_heads x num_steps x ceil(sk / 128) x (4 + max_sq / 128)

    and the fully unrolled kernel costs ceil(sq / max_sq) x per_q_block. The budget allows
    budget_blocks = _MAX_STATIC_ATTN_INSTRUCTIONS // per_q_block statically emitted Q-blocks.

    Selection. If all ceil(sq / max_sq) Q-blocks fit budget_blocks, return None: the kernel stays
    fully unrolled. Otherwise the rolled path (_attention_const_max_hbm_acc) emits G full blocks
    once, inside the nl.fori_loop body, and emits the leftover full blocks (num_full % G) and the
    partial tail block statically after the loop. Return the largest G for which
    G + num_full % G + tail <= budget_blocks and the loop runs at least 2 iterations
    (num_full // G >= 2). If no G fits, return 1, the smallest body.

    Wan2.2 values (10 heads, CP8; max_sq is 512 on trn2 and 768 on trn3, see _max_sq_block_size):
        720p, 9450 tokens/rank: ~0.77M (trn3) and ~0.90M (trn2) unrolled, so None.
        1080p, 21105 tokens/rank, trn3: 28 blocks x 132K = ~3.7M, budget 7 blocks, G = 3
            (3 + 0 + 1 = 4 blocks emitted, ~0.53M, 9 iterations).
        1080p, 21105 tokens/rank, trn2: 42 blocks x 106K = ~4.4M, budget 9 blocks, G = 5
            (5 + 1 + 1 = 7 blocks emitted, ~0.74M, 8 iterations).
    """
    max_sq = _max_sq_block_size()
    num_q_blocks = (sq + max_sq - 1) // max_sq
    kv_tiles = (sk + _P - 1) // _P
    per_q_block = num_heads * num_steps * kv_tiles * (_INSTRUCTIONS_PER_KV_TILE + max_sq // _O_TILE_SIZE)
    budget_blocks = max(1, _MAX_STATIC_ATTN_INSTRUCTIONS // per_q_block)
    if num_q_blocks <= budget_blocks:
        return None
    num_full = sq // max_sq
    has_tail = 1 if sq % max_sq else 0
    for group in range(min(budget_blocks, num_full), 0, -1):
        if group + num_full % group + has_tail <= budget_blocks and num_full // group >= 2:
            return group
    return 1


def _attention_const_max_partial(
    q_hbm,
    k_hbm,
    v_hbm,
    o_aug_acc_sb,
    c_row_hbm,
    softmax_scale=None,
    aug_col_base=0,
    is_accumulate_step=False,
    kv_pad_rows=0,
):
    """Partial (unnormalized) attention folding into a resident SBUF accumulator.

    Runs the K-stationary MM1 → exp(scores − c) → MM2([V|1]) pipeline and folds each output
    tile's [Σ P·V | Σ P·1] straight from PSUM into the caller's resident SBUF accumulator —
    is_accumulate_step selects init (ring step 0) vs += (later steps), so there is no HBM
    round-trip and no separate cross-step reduction pass. Does NO LNC sharding of its own: the
    caller passes the head slice this shard owns.

    Args:
        q_hbm: [N, d, Sq] bf16 @ HBM.
        k_hbm: [N, d, Sk] bf16 @ HBM (arbitrary Sk; a trailing sub-128 group is handled).
        v_hbm: [N, Sk, d] bf16 @ HBM.
        o_aug_acc_sb: resident SBUF accumulator this step folds into. Layout mirrors the PSUM
            source: each (head, group) tile occupies (d+1) contiguous columns — d output columns
            followed by 1 sum column.
        c_row_hbm: [N, Sq] bf16 @ HBM holding the per-ROW bound c_i = softmax_scale·‖q_i‖·max_j‖k_j‖
            in natural token order. Each Q-block loads its contiguous [1, sq_block] slice onto the
            free axis and broadcasts it across the 128 partitions.
        softmax_scale: scaling factor for Q. Defaults to 1/sqrt(d).
        aug_col_base: base column of this call's first (head, group) tile (stride d+1 between tiles).
        is_accumulate_step: True folds (+=); False initializes (first ring step).
        kv_pad_rows: Number of trailing Sk tokens that are caller-supplied ZERO padding (used when
            the caller rounded Sk up to a 128-multiple to keep the KV loop uniform). Those rows are
            masked out of the sums exactly; 0 means Sk is entirely real tokens.

    KV split. Sk is cut into up to three sources that the pipeline consumes in order:

    * blocks (``k_stream``/``v_stream``): the first ``sk_block_rows`` rows, streamed in blocks of
      ``_SK_BLOCK`` = 2048 rows (16 tiles of 128) with double-buffered prefetch;
    * leftover tiles (``k_leftover``/``v_leftover``): the whole 128-row tiles after the last full
      block, loaded once as one multi-tile grid (``sk_leftover_rows`` rows);
    * tail (``k_tail``/``v_tail``): the final ``Sk % 128`` rows as one partial tile.

    A neurotile block stream sizes its rotating buffers from its first block, so it can carry
    either several full 16-tile blocks or a single block of any size up to 16 tiles, but not
    full blocks followed by a shorter one (the DMA into the 16-tile buffer then fails to trace:
    "dma_copy requires src and dst to have the same number of elements"). Hence the rule
    ``sk_block_rows = sk_whole`` when all whole tiles fit in one block (``sk_whole <= _SK_BLOCK``)
    and ``sk_block_rows = (sk_whole // _SK_BLOCK) * _SK_BLOCK`` otherwise; the rest of the whole
    tiles become the leftover source. Examples, with ``sk_whole = (Sk // 128) * 128``:

    * Sk = 1000: 7 whole tiles fit one block, so the stream carries one 7-tile block, there are
      no leftover tiles, and the tail has 104 rows (7 + 1 = 8 tiles).
    * Sk = 2176 (FLUX.1 2048 px at CP8, 17 tiles): one 16-tile block, one leftover tile, no tail.
    * Sk = 4700: 36 whole tiles do not fit one block, so the stream carries two 16-tile blocks
      (4096 rows), the leftover source holds 4 tiles (512 rows), and the tail has 92 rows
      (16 + 16 + 4 + 1 = 37 tiles).

    Every Sk that traced before this split (at most 16 tiles, or a multiple of 16) keeps the same
    sources: ``sk_leftover_rows`` is 0 for it.

    Pseudocode:
        for each head:
            for each Q-block:
                Q = load(q_hbm[head, Q-block]) * softmax_scale
                c = load(c_row_hbm[head, Q-block])       # per-row bound, broadcast over partitions
                for each KV-block:
                    for each kv tile:
                        S = Q @ K_tile.T                 # MM1: scores
                        P = exp(S - c)                   # evict + subtract + exp
                        O_aug += P @ [V_tile | 1]        # MM2: output + sum_exp
                fold O_aug into o_aug_acc_sb
    """
    Nq, d, Sq = q_hbm.shape
    Sk = k_hbm.shape[2]
    softmax_scale = d ** (-0.5) if softmax_scale == None else softmax_scale
    # Groups (128-row Sq output tiles) per head — the accumulator stride between heads.
    groups_per_head = (Sq + _O_TILE_SIZE - 1) // _O_TILE_SIZE

    kernel_assert(nl.tile_size.psum_num_banks >= 6, f"need at least 6 PSUM banks, have {nl.tile_size.psum_num_banks}")
    max_o_aug_banks = _max_o_aug_banks()
    max_sq_block_size = _max_sq_block_size()

    sk_whole = (Sk // _P) * _P
    r_tail = Sk % _P
    total_kv_tiles = (Sk + _P - 1) // _P
    sk_block_rows = sk_whole if sk_whole <= _SK_BLOCK else (sk_whole // _SK_BLOCK) * _SK_BLOCK
    sk_leftover_rows = sk_whole - sk_block_rows

    for head_idx in range(Nq):
        # Q is [d, Sq] per head, tiled along Sq at max_sq_block_size; each Q block loaded once and
        # kept resident while all KV blocks stream through it.
        q_block_grid = nt.tiles(q_hbm[head_idx], tile_size=(_P, max_sq_block_size))

        for q_block_idx in range(q_block_grid.shape[1]):
            # Load Q block into SBUF — stays resident for all KV iterations. Q is pre-scaled here
            # rather than folding softmax_scale into the exp, because the bound is
            # c = softmax_scale·‖q‖·max‖k‖: the scores it is subtracted from must already carry
            # the scale.
            q_loaded = q_block_grid[0, q_block_idx].load()
            q_block = q_loaded.data
            nisa.tensor_scalar(dst=q_block, data=q_block, op0=nl.multiply, operand0=softmax_scale)

            sq_block_size = q_loaded.element_shape[1]

            # c_row_hbm holds c_i in natural token order, so this Q-block's contiguous
            # [1, sq_block] slice IS c_i along the score tile's free (Sq) axis.
            # Replicate it down all 128 partitions with ONE partition-stride-0 DMA.
            #
            # Every partition must hold the IDENTICAL c for a query column: softmax is invariant to c only
            # then, since a column whose partitions disagree scales its Sk terms inconsistently and the
            # factor stops cancelling in o = ΣPV/ΣP. 128 reads of the same immutable HBM row give exactly
            # that, and they cost no Vector time. Deriving all 128 partitions from one loaded row with a
            # stream shuffle instead is the same guarantee, but it put ~3.4us of serialized Vector work on
            # every Q-block boundary AHEAD of the Q pre-scale in the in-order Vector queue — and the
            # pre-scale gates this block's first MM1, so the Tensor engine waited out both.
            sq0 = q_block_idx * max_sq_block_size
            per_row_bias = nl.ndarray(shape=(_P, sq_block_size), dtype=c_row_hbm.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=per_row_bias,
                src=c_row_hbm[head_idx].ap(pattern=[[0, _P], [1, sq_block_size]], offset=sq0),
            )

            # Allocate output accumulator in PSUM — one tile per Sq output slice, each on its own
            # bank. Accumulate across all KV blocks, then fold into the SBUF accumulator.
            num_output_tiles = (sq_block_size + _O_TILE_SIZE - 1) // _O_TILE_SIZE
            kernel_assert(
                num_output_tiles <= max_o_aug_banks,
                "Q block too large: O_aug tiles exceed available PSUM banks.",
            )
            # Output accumulator width is d+1: columns 0..d-1 hold the weighted output (P @ V), and
            # column d accumulates sum_exp (P @ 1) for the caller's normalize.
            o_aug_psums = nt.psum_pool(
                tile_size=(_O_TILE_SIZE, d + 1),
                element_shape=(sq_block_size, d + 1),
            )
            # Continuous software pipeline across all KV tiles keeps TensorE fed (no per-block
            # boundary bubbles). nt.blocks handles a non-_SK_BLOCK-multiple prefix: the last block
            # has fewer tiles.
            k_stream = None
            v_stream = None
            if sk_block_rows > 0:
                k_view = nt.blocks(
                    k_hbm[head_idx, :, :sk_block_rows],
                    tile_size=(_P, d),
                    block_size=(1, _KV_TILES_PER_BLOCK),
                )
                v_view = nt.blocks(
                    v_hbm[head_idx, :sk_block_rows, :],
                    tile_size=(_P, d),
                    block_size=(_KV_TILES_PER_BLOCK, 1),
                )
                k_stream = k_view[0, :].stream(buffer_count=2)
                v_stream = v_view[:, 0].stream(buffer_count=2)

            k_leftover = None
            v_leftover = None
            if sk_leftover_rows > 0:
                k_leftover = nt.tiles(k_hbm[head_idx, :, sk_block_rows:sk_whole], tile_size=(_P, d))[
                    0, :
                ].load()
                v_leftover = nt.tiles(v_hbm[head_idx, sk_block_rows:sk_whole, :], tile_size=(_P, d))[
                    :, 0
                ].load()

            # The sub-128 tail as its own single partial tile — a 1-tile grid passes the neurotile
            # load, and it indexes exactly like a block-stream tile grid.
            k_tail = None
            v_tail = None
            if r_tail > 0:
                k_tail = nt.tiles(k_hbm[head_idx, :, sk_whole:Sk], tile_size=(_P, d))[0, :].load()
                v_tail = nt.tiles(v_hbm[head_idx, sk_whole:Sk, :], tile_size=(_P, d))[:, 0].load()

            _process_kv_tiles_pipelined(
                k_stream=k_stream,
                v_stream=v_stream,
                q_block=q_block,
                o_aug_psums=o_aug_psums,
                per_row_bias=per_row_bias,
                k_leftover=k_leftover,
                v_leftover=v_leftover,
                leftover_tiles=sk_leftover_rows // _P,
                k_tail=k_tail,
                v_tail=v_tail,
                total_kv_tiles=total_kv_tiles,
                kv_pad_rows=kv_pad_rows,
            )

            # Fold each output tile's accumulated o_aug = [Σ P·V | Σ P·1] straight from PSUM into the
            # resident SBUF accumulator (init on the first ring step, += after).
            o_block_start = q_block_idx * (max_sq_block_size // _O_TILE_SIZE)
            for output_tile_idx in range(num_output_tiles):
                g = head_idx * groups_per_head + (o_block_start + output_tile_idx)
                _accumulate_partial(
                    o_aug_psums[output_tile_idx].data,
                    o_aug_acc_sb,
                    aug_col_base + g * (d + 1),
                    is_accumulate_step,
                )


def _process_kv_tiles_pipelined(
    k_stream: Optional[nt.BlockStream],
    v_stream: Optional[nt.BlockStream],
    q_block: nl.NkiTensor,
    o_aug_psums: nt.NDSlice,
    per_row_bias: nl.NkiTensor,
    total_kv_tiles: int,
    k_leftover: Optional[nt.NDSlice] = None,
    v_leftover: Optional[nt.NDSlice] = None,
    leftover_tiles: int = 0,
    k_tail: Optional[nt.NDSlice] = None,
    v_tail: Optional[nt.NDSlice] = None,
    kv_pad_rows: int = 0,
    inside_fori_loop: bool = False,
) -> None:
    """Pipelined MM1→exp→MM2 across all KV tiles with a prologue/epilogue split.

    The bound c_i varies along the free (Sq) axis, so it cannot ride the exp's per-partition
    max_value. It is subtracted explicitly (Vector) into an fp16 tile, which the exp then reads.
    The exp is issued via nisa.activation (Scalar engine) so the Vector engine does only the
    subtract and the two pipeline across tiles; the [V|1] copy goes to whichever of the two is
    lighter (see copy_engine below), and all of them pipeline against the Tensor engine's MMs.

    Pipeline (P = prologue_depth):
        Prologue  (kv_tile_idx 0..P-1):        v_copy, MM1, exp
        Steady-state (kv_tile_idx P..total-1):  v_copy, MM1, exp, MM2(i-P)
        Epilogue  (drain last P):               MM2

    A trailing partial tile (the peeled sub-128 KV tail) has r < 128 valid Sk rows. Every
    Sk-indexed extent — the V copy, MM1 output partitions, exp, and the MM2 contraction on both
    operands — is clamped to that tile's real row count r (== v_block[i, 0].data.shape[0]), so
    padded positions contribute nothing to Σ P·V or Σ P·1 and no explicit tail mask is needed.

    Args:
        k_stream/v_stream: block streams over the first sk_block_rows Sk rows: full 16-tile
            blocks, or one block of at most 16 tiles when all whole 128-row tiles fit in one block
            (see the KV split in _attention_const_max_partial). None when Sk < 128.
        k_leftover/v_leftover: Optional pre-loaded multi-tile grids holding the whole 128-row
            tiles after the last streamed block and before the sub-128 tail
            (K (128, leftover_tiles*128), V (leftover_tiles*128, d)). None when the streamed
            blocks reach the tail.
        leftover_tiles: Tiles in k_leftover/v_leftover; 0 when they are None.
        per_row_bias: [_P, sq_block_size] per-row bound c_i replicated down all 128 partitions
            (uniform per free column = per query row).
        k_tail/v_tail: Optional pre-loaded single-tile grids holding the sub-128 Sk tail
            (K tile (128, r_tail), V tile (r_tail, d)). Consumed as the pipeline's last tile.
        kv_pad_rows: Trailing partitions of the LAST tile that are caller-supplied ZERO padding
            (Sk already rounded up to a 128-multiple). Their ones-column is zeroed so they add
            nothing to Σ P·1; Σ P·V is already zero because the padded V tokens are zero.
        total_kv_tiles: ceil(Sk / 128) — the exact flat tile count, including the tail tile. Sets
            the pipeline depth and the epilogue drain range; the last prefix block may be partial,
            so this cannot be derived from num_blocks.
        inside_fori_loop: True when called from an nl.fori_loop body. There the K/V stream loads
            use hwdge, which DMAs inside the loop require, and the pipeline is not wrapped in
            nl.no_reorder(), which is not allowed inside the loop. Otherwise the loads use the
            default DGE mode, and each block's tiles and the epilogue run under nl.no_reorder().

    Sources are visited in KV order: source indices 0..num_blocks-1 are the streamed blocks,
    then the leftover grid when leftover_tiles > 0, then the tail when k_tail is not None. The
    tail is therefore the last source only when it exists: with leftover tiles and no tail (any
    128-multiple Sk above 16 tiles that is not a multiple of 16, such as the ring driver's padded
    2176), the last source is the leftover grid, which is why is_tail also checks k_tail.
    """
    d = _P
    num_blocks = k_stream.count if k_stream != None else 0
    sq_block_size = q_block.shape[1]
    num_output_tiles = o_aug_psums.shape[0]

    scores_dtype = nl.bfloat16 if nisa.get_nc_version() >= nisa.nc_version.gen4 else nl.float32
    # V-augmentation copy engine: route to whichever engine is NOT running exp so the copy overlaps.
    # NEVER GpSimd. The compiler triggers the ring's KV collective-permute from GpSimd, and that queue
    # is in-order, so a per-tile copy stream there defers every permute trigger until the previous ring
    # step's copies drain: the driver's kv_prefetch_depth hoist stops taking effect on hardware and the
    # hops go one-per-step (measured ~400-620us of channel idle per hop instead of ~2us). Where the
    # collective cannot hide behind per-core compute that directly stalls the next step's MM1: moving
    # this copy off GpSimd is worth +3.4 MFU points at trn3 cp4 lnc1 (60.9 -> 64.3) and +5.9 at cp4
    # lnc2 (54.1 -> 60.0) on WAN 480p.
    # The subtract runs on Vector and the exp on Scalar, so pick the lighter of those two. The
    # subtract's cost tracks the `shifted` precision, and `shifted` is fp16, which leaves Vector the
    # lighter of the pair -- so the copy goes there. It is ~3.4x cheaper there than on GpSimd.
    copy_engine = nisa.engine.vector

    # Number of MM1+exp tiles computed before first MM2 fires. Gives TensorE
    # a buffer of ready probabilities so it can always issue MM2 without waiting. Bounded by the
    # exact tile count (not num_blocks * tiles-per-block, which over-counts a partial last block
    # and would make the epilogue drain tiles that were never issued).
    prologue_depth = min(8, total_kv_tiles)

    # Rotating V_aug buffers: need prologue_depth+1 (v_copy runs at MM1 time,
    # MM2 consumes prologue_depth iterations later; +1 to avoid WAR)
    num_v_aug_buffers = prologue_depth + 1
    v_aug_all = nl.ndarray(shape=(_P, num_v_aug_buffers, d + 1), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(v_aug_all[:, :, d : d + 1], 1.0)

    num_prob_buffers = prologue_depth + 1
    probs_all = nl.ndarray(shape=(_P, num_prob_buffers, sq_block_size), dtype=nl.bfloat16, buffer=nl.sbuf)

    # `shifted = scores - c_i`, consumed by the exp. fp16 (not bf16) because the exp needs ABSOLUTE
    # precision in the exponent: bf16's 8-bit mantissa leaves ~0.004*|shifted| (~4% on a prob) and
    # fails accuracy, while fp16's 11 bits leave ~0.0005*|shifted| — below the error the bf16 PSUM
    # scores already carry, and finest exactly where |shifted| ~ 0 (the probs that dominate the sum).
    # Halving this tile vs fp32 also halves the subtract's write and the exp's read, the SBUF traffic
    # that was stretching both past 1 cycle/element and slowing the Tensor engine with it.
    # Lifetime is one tile (subtract → exp), so double-buffer — lets subtract(i+1) overlap exp(i).
    # (Deepening this to one buffer per in-flight tile was tried to chase a head-0 miscompute: it made
    # the failure MORE frequent and total rather than partial, so the defect is not this aliasing.)
    num_shifted_buffers = 2
    shifted_all = nl.ndarray(
        shape=(_P, num_shifted_buffers, sq_block_size),
        dtype=nl.float16,
        buffer=nl.sbuf,
    )

    # Per-tile valid-Sk-row slice, recorded as each tile is issued so the deferred MM2 (fired
    # prologue_depth iterations later) and the epilogue drain can reproduce that tile's extent.
    # Every whole-128 prefix tile gets the full _P-partition slice (the extent of every tile `rows`
    # indexes), so the prefix emits exactly the accesses it would for a 128-multiple Sk; only the
    # peeled tail tile is clamped.
    full_rows = slice(0, _P)
    tile_rows = []

    kv_tile_idx = 0  # flat index across all KV tiles

    num_leftover_sources = 1 if leftover_tiles > 0 else 0
    num_sources = num_blocks + num_leftover_sources + (1 if k_tail is not None else 0)
    dge_mode = nisa.dge_mode.hwdge if inside_fori_loop else None

    def _issue_tiles(k_block, v_block, tiles_in_block, is_tail, kv_tile_idx):
        """v_copy, MM1, exp (and the deferred MM2) for one source's tiles; returns the next flat index."""
        for local_tile in range(tiles_in_block):
            v_buf = kv_tile_idx % num_v_aug_buffers
            prob_buf = kv_tile_idx % num_prob_buffers
            # Rows valid in this tile: all 128 for a prefix tile, r_tail for the tail tile.
            rows = slice(0, v_block[local_tile, 0].data.shape[0]) if is_tail else full_rows
            tile_rows.append(rows)

            # v_copy(i) — capture V tile now while block buffer is valid
            nisa.tensor_copy(
                dst=v_aug_all[rows, v_buf, :d],
                src=v_block[local_tile, 0].data,
                engine=copy_engine,
            )

            # Padded-KV mask. When the caller padded Sk up to a 128-multiple with ZEROED tokens,
            # the last tile's trailing kv_pad_rows partitions are padding. Their V data columns
            # are zero, so Σ P·V is already exact — but the [V|1] ones column would still fold
            # those rows' exp into Σ P·1. Zeroing the ones column there makes BOTH terms exactly
            # zero, so no masked bias is needed. Both memsets start at partition 0 (a non-zero
            # partition start offset is rejected), hence zero-the-column-then-rewrite-the-real-rows
            # rather than memsetting the pad rows.
            if kv_pad_rows > 0 and kv_tile_idx == total_kv_tiles - 1:
                nisa.memset(v_aug_all[:, v_buf, d : d + 1], 0.0)
                nisa.memset(v_aug_all[0 : _P - kv_pad_rows, v_buf, d : d + 1], 1.0)

            # MM1(i) — the K tile's free extent is this tile's row count, so the matmul
            # writes exactly that many score partitions.
            scores = nl.ndarray(shape=(_P, sq_block_size), dtype=scores_dtype, buffer=nl.psum)
            nisa.nc_matmul(dst=scores[rows, :], stationary=k_block[local_tile].data, moving=q_block)

            # exp(i) — subtract the per-row bound, then exponentiate.
            # c_i varies along the free (Sq) axis, so it can't ride the exp's per-partition
            # max_value. Subtract it explicitly (Vector; scores PSUM - c SBUF into an fp16 tile,
            # see shifted_all), then exp on the SCALAR engine (nisa.activation) so the Vector
            # engine does ONLY the subtract and the two pipeline across tiles (the [V|1] copy
            # rides the lighter of the two). Running exp on Vector too is ~1.5x slower.
            # (Folding -c into the score bank ahead of an accumulating MM1 would remove this
            # pass, but nc_matmul cannot target an fp16 PSUM dst and a bf16 bank would round
            # scores-c, whose absolute error ~0.004*c is what the per-row bound exists to avoid.)
            shifted_buf = kv_tile_idx % num_shifted_buffers
            nisa.tensor_tensor(
                dst=shifted_all[rows, shifted_buf, :],
                data1=scores[rows, :],
                data2=per_row_bias[rows, :],
                op=nl.subtract,
                engine=nisa.engine.vector,
            )
            nisa.activation(
                dst=probs_all[rows, prob_buf, :],
                data=shifted_all[rows, shifted_buf, :],
                op=nl.exp,
                scale=1.0,
                bias=0.0,
            )

            # MM2(i-P) — only after prologue is filled
            prologue_filled = kv_tile_idx >= prologue_depth
            if prologue_filled:
                consume_idx = kv_tile_idx - prologue_depth
                _issue_mm2(
                    o_aug_psums=o_aug_psums,
                    probs_all=probs_all,
                    v_aug_all=v_aug_all,
                    tile_idx=consume_idx,
                    rows=tile_rows[consume_idx],
                    num_prob_buffers=num_prob_buffers,
                    num_v_aug_buffers=num_v_aug_buffers,
                    num_output_tiles=num_output_tiles,
                )

            kv_tile_idx += 1
        return kv_tile_idx

    for source_idx in range(num_sources):
        is_tail = k_tail is not None and source_idx == num_sources - 1
        if is_tail:
            k_block = k_tail
            v_block = v_tail
            tiles_in_block = 1
        elif source_idx >= num_blocks:
            k_block = k_leftover
            v_block = v_leftover
            tiles_in_block = leftover_tiles
        else:
            k_block = k_stream.load(source_idx, dge_mode=dge_mode)[0]
            v_block = v_stream.load(source_idx, dge_mode=dge_mode)
            tiles_in_block = k_block.shape[0]

        if inside_fori_loop:
            kv_tile_idx = _issue_tiles(k_block, v_block, tiles_in_block, is_tail, kv_tile_idx)
        else:
            with nl.no_reorder():
                kv_tile_idx = _issue_tiles(k_block, v_block, tiles_in_block, is_tail, kv_tile_idx)

    # === Epilogue: drain remaining prologue_depth MM2s ===
    def _drain_epilogue():
        for i in range(prologue_depth):
            drain_idx = total_kv_tiles - prologue_depth + i
            _issue_mm2(
                o_aug_psums=o_aug_psums,
                probs_all=probs_all,
                v_aug_all=v_aug_all,
                tile_idx=drain_idx,
                rows=tile_rows[drain_idx],
                num_prob_buffers=num_prob_buffers,
                num_v_aug_buffers=num_v_aug_buffers,
                num_output_tiles=num_output_tiles,
            )

    if inside_fori_loop:
        _drain_epilogue()
    else:
        with nl.no_reorder():
            _drain_epilogue()


def _issue_mm2(
    o_aug_psums,
    probs_all,
    v_aug_all,
    tile_idx,
    rows,
    num_prob_buffers,
    num_v_aug_buffers,
    num_output_tiles,
) -> None:
    """MM2 for one KV tile: P @ [V|1] accumulated into the O_aug PSUM tiles (tiled along Sq).

    Both operands clamp their contraction (Sk) to `rows`, so a peeled sub-128 tail tile's padded
    rows add nothing to Σ P·V or Σ P·1. Shared by the steady-state and the epilogue drain, which
    issue the same MM2 for different tile indices.
    """
    prob_buf = tile_idx % num_prob_buffers
    v_aug = v_aug_all[:, tile_idx % num_v_aug_buffers, :]
    should_accumulate = tile_idx != 0
    for ot_idx in range(num_output_tiles):
        sq_tile_size = o_aug_psums[ot_idx].data.shape[0]
        nisa.nc_matmul(
            dst=o_aug_psums[ot_idx].data,
            stationary=probs_all[rows, prob_buf, nl.ds(ot_idx * _O_TILE_SIZE, sq_tile_size)],
            moving=v_aug[rows, :],
            accumulate=should_accumulate,
        )


def _accumulate_partial(o_aug_psum, o_aug_acc_sb, aug_col, is_accumulate_step):
    """Fold one output tile's [Σ P·V | Σ P·1] from PSUM into a resident SBUF accumulator.

    Replaces the store-to-HBM + reload-and-add cross-step reduction with a single PSUM→SBUF op per
    tile: init (tensor_copy) on the first ring step, add (tensor_tensor) after.

    o_aug_psum: [sq_tile_size, d+1] PSUM — cols 0..d-1 are Σ P·V, col d is Σ P·1 (sum_exp).
    o_aug_acc_sb: [_P, ...] SBUF accumulator (this run's view); this tile lands at
        o_aug_acc_sb[:, aug_col:aug_col+d+1] (d output cols followed by 1 sum col).
    is_accumulate_step: True => += (fold), False => = (initialize).
    """
    sq_tile_size = o_aug_psum.shape[0]
    d = o_aug_psum.shape[1] - 1
    aug_dst = o_aug_acc_sb[:sq_tile_size, aug_col : aug_col + d + 1]
    if is_accumulate_step:
        nisa.tensor_tensor(dst=aug_dst, data1=aug_dst, data2=o_aug_psum, op=nl.add)
    else:
        nisa.tensor_copy(dst=aug_dst, src=o_aug_psum)


def _attention_const_max_hbm_acc(
    q_root,
    k_root,
    v_root,
    acc_root,
    c_root,
    o_root,
    slots,
    q_group,
    softmax_scale=None,
    is_first_step=True,
    is_last_step=True,
    kv_pad_rows=0,
    sum_clamp=1e-37,
):
    """Partial attention with the Q-block loop rolled and the cross-step accumulator in HBM.

    Same math as _attention_const_max_partial, for shapes whose fully unrolled trace would overflow
    the BIR body (see choose_q_group). The nl.fori_loop body is traced once, so a Q-block's
    position is a runtime value (the loop register). The per-block DMAs take it as a scalar offset,
    and the compute reads fixed SBUF tiles. Each Q-block's [Σ P·V | Σ P·1] accumulator persists
    across ring steps in HBM, and every fold DMAs it by that offset into a fixed SBUF staging tile
    and back: ring step 0 writes it, later steps read-add-write it, and the last step normalizes it
    and writes the output instead (so there is no separate normalize pass). The accumulators are in
    HBM rather than a resident SBUF region because at 1080p they are ~54 MB per core, which would
    need head chunking to fit in SBUF.

    Per head, the full Q-blocks run as nl.fori_loop iterations of ``q_group`` blocks each, unrolled
    inside the body; the leftover full blocks and the partial tail block are emitted statically.
    The loop variable is the group's first query row, which every per-block DMA takes as its
    scalar offset. Every tensor is a ROOT private-HBM tensor indexed by the caller's per-core head
    slot: DMAs index the root (so indirect_dim is the root's row-major dimension), and nothing the
    loop body reads is shared HBM (a shared read inside the loop needs a core barrier there).

    Args:
        q_root: [H, d, Sq] bf16 @ private HBM.
        k_root: [H, d, Sk] bf16 @ private HBM, Sk a multiple of 128 (the ring driver pads it).
        v_root: [H, Sk, d] bf16 @ private HBM.
        acc_root: [H, S_rows, d+1] fp32 @ private HBM, S_rows >= Sq; row i holds query i's
            [Σ P·V | Σ P·1].
        c_root: [H, S_rows] per-row softmax bound (see _attention_const_max_partial).
        o_root: [H, Sq, d] output, written on the last step.
        slots: head slots (indices into dim 0 of every root tensor) to process.
        q_group: full Q-blocks per loop iteration (from choose_q_group).
        softmax_scale: scaling factor for Q. Defaults to 1/sqrt(d).
        is_first_step / is_last_step: position in the ring; both True for a single-step ring.
        kv_pad_rows: trailing zero-padding tokens in Sk (see _attention_const_max_partial).
        sum_clamp: floor on Σ P·1 before the reciprocal.
    """
    _, d, Sq = q_root.shape
    Sk = k_root.shape[2]
    softmax_scale = d ** (-0.5) if softmax_scale == None else softmax_scale
    kernel_assert(Sk % _P == 0, f"rolled attention expects Sk padded to a multiple of {_P}, got {Sk}")
    max_sq = _max_sq_block_size()
    num_full = Sq // max_sq
    tail = Sq - num_full * max_sq
    num_iters = num_full // q_group
    rolled_blocks = num_iters * q_group if num_iters >= 2 else 0
    shape = _HbmAccShape(d=d, sq=Sq, sk=Sk, s_rows=acc_root.shape[1])

    # Arguments every Q-block shares; the per-block ones are slot, start, size and the register.
    common = (q_root, k_root, v_root, acc_root, c_root, o_root, shape)
    flags = (softmax_scale, is_first_step, is_last_step, kv_pad_rows, sum_clamp)
    for slot in slots:
        if rolled_blocks > 0:

            def _group_body(sq_dyn, slot=slot):
                for local_block in range(q_group):
                    _hbm_acc_q_block(*common, slot, local_block * max_sq, max_sq, sq_dyn, *flags)

            nl.fori_loop(0, rolled_blocks * max_sq, _group_body, step=q_group * max_sq)
        for block_idx in range(rolled_blocks, num_full):
            _hbm_acc_q_block(*common, slot, block_idx * max_sq, max_sq, None, *flags)
        if tail > 0:
            _hbm_acc_q_block(*common, slot, num_full * max_sq, tail, None, *flags)


class _HbmAccShape(nl.NKIObject):
    """Root-tensor extents shared by every Q-block of one _attention_const_max_hbm_acc call."""

    def __init__(self, d, sq, sk, s_rows):
        self.d = d
        self.sq = sq
        self.sk = sk
        self.s_rows = s_rows


def _hbm_acc_q_block(
    q_root,
    k_root,
    v_root,
    acc_root,
    c_root,
    o_root,
    shape,
    slot,
    sq_start,
    sq_size,
    sq_dyn,
    softmax_scale,
    is_first_step,
    is_last_step,
    kv_pad_rows,
    sum_clamp,
):
    """One Q-block of _attention_const_max_hbm_acc: MM1 → exp → MM2 over all KV tiles, then fold.

    Covers query rows sq_dyn + sq_start .. + sq_size of head ``slot``, where sq_dyn is the
    fori_loop register (the group's first query row) or None for a statically placed block. Every
    query-indexed DMA adds it as a scalar offset along the root tensor's query-row dimension
    (stride 1 for q and c, d+1 for the accumulator, d for the output); nothing else depends on it.
    """
    d, Sq, Sk, s_rows = shape.d, shape.sq, shape.sk, shape.s_rows
    hwdge = nisa.dge_mode.hwdge

    q_block = nl.ndarray(shape=(d, sq_size), dtype=q_root.dtype, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=q_block,
        src=_row_ap(sq_dyn, q_root, [[Sq, d], [1, sq_size]], slot * d * Sq + sq_start, 2),
        dge_mode=hwdge,
    )
    nisa.tensor_scalar(dst=q_block, data=q_block, op0=nl.multiply, operand0=softmax_scale)

    # Per-row bound c_i on the free (Sq) axis, replicated down all 128 partitions by a
    # partition-stride-0 DMA (see _attention_const_max_partial).
    per_row_bias = nl.ndarray(shape=(_P, sq_size), dtype=c_root.dtype, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=per_row_bias,
        src=_row_ap(sq_dyn, c_root, [[0, _P], [1, sq_size]], slot * s_rows + sq_start, 1),
        dge_mode=hwdge,
    )

    num_output_tiles = (sq_size + _O_TILE_SIZE - 1) // _O_TILE_SIZE
    o_aug_psums = nt.psum_pool(tile_size=(_O_TILE_SIZE, d + 1), element_shape=(sq_size, d + 1))
    k_view = nt.blocks(k_root[slot], tile_size=(_P, d), block_size=(1, _KV_TILES_PER_BLOCK))
    v_view = nt.blocks(v_root[slot], tile_size=(_P, d), block_size=(_KV_TILES_PER_BLOCK, 1))
    _process_kv_tiles_pipelined(
        k_stream=k_view[0, :].stream(buffer_count=2),
        v_stream=v_view[:, 0].stream(buffer_count=2),
        q_block=q_block,
        o_aug_psums=o_aug_psums,
        per_row_bias=per_row_bias,
        total_kv_tiles=Sk // _P,
        kv_pad_rows=kv_pad_rows,
        inside_fori_loop=sq_dyn is not None,
    )

    for output_tile_idx in range(num_output_tiles):
        o_aug_psum = o_aug_psums[output_tile_idx].data  # [rows, d+1]: Σ P·V cols, then Σ P·1
        rows = o_aug_psum.shape[0]
        row0 = sq_start + output_tile_idx * _O_TILE_SIZE
        acc_rows = _row_ap(sq_dyn, acc_root, [[d + 1, rows], [1, d + 1]], (slot * s_rows + row0) * (d + 1), 1)
        if is_first_step and is_last_step:
            folded = o_aug_psum
        else:
            stage = nl.ndarray(shape=(_P, d + 1), dtype=nl.float32, buffer=nl.sbuf)
            if is_first_step:
                nisa.tensor_copy(dst=stage[:rows, :], src=o_aug_psum)
            else:
                nisa.dma_copy(dst=stage[:rows, :], src=acc_rows, dge_mode=hwdge)
                nisa.tensor_tensor(dst=stage[:rows, :], data1=stage[:rows, :], data2=o_aug_psum, op=nl.add)
            folded = stage[:rows, :]

        if is_last_step:
            o_rows = _row_ap(sq_dyn, o_root, [[d, rows], [1, d]], (slot * Sq + row0) * d, 1)
            _normalize_rows_to_hbm(folded, rows, o_rows, d, sum_clamp)
        else:
            nisa.dma_copy(dst=acc_rows, src=folded, dge_mode=hwdge)


def _row_ap(sq_dyn, root, pattern, offset, dim):
    """Access pattern on a root HBM tensor, shifted by the fori_loop register ``sq_dyn`` (query rows,
    scaled by the stride of the root's dimension ``dim``) when the block is inside the loop."""
    if sq_dyn is None:
        return root.ap(pattern=pattern, offset=offset)
    return root.ap(pattern=pattern, offset=offset, scalar_offset=sq_dyn, indirect_dim=dim)


def _normalize_rows_to_hbm(acc_rows, rows, o_rows, d, sum_clamp):
    """o = Σ P·V / max(Σ P·1, sum_clamp) for one output tile, stored to o_rows (HBM)."""
    sum_clamped = nl.ndarray(shape=(_P, 1), dtype=nl.float32, buffer=nl.sbuf)
    inv_sum = nl.ndarray(shape=(_P, 1), dtype=nl.float32, buffer=nl.sbuf)
    o_tile = nl.ndarray(shape=(_P, d), dtype=o_rows.dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=sum_clamped[:rows, :], data=acc_rows[:, d : d + 1], op0=nl.maximum, operand0=sum_clamp)
    nisa.reciprocal(dst=inv_sum[:rows, :], data=sum_clamped[:rows, :])
    nisa.tensor_scalar(dst=o_tile[:rows, :], data=acc_rows[:, :d], op0=nl.multiply, operand0=inv_sum[:rows, :])
    nisa.dma_copy(dst=o_rows, src=o_tile[:rows, :], dge_mode=nisa.dge_mode.hwdge)
