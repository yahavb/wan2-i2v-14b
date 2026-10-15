# Self-attention in Wan2.2 I2V 14B: legacy kernel vs attention_cte

Run: `exp/staging-attn`, 368x480, 21 frames, 10 steps, batched CFG, TP=4 SP=2, trn2 (LNC2).
Profiles: Neuron Explorer, one captured execution of each kernel's NEFF.

## 1. What this attention is

Self-attention is the step inside each of the DiT's 40 transformer blocks where every
video token (4,140 of them: 6 latent frames x 23 x 30 patches) looks at every other token
and decides what to take from it. It runs 40 blocks x 10 denoising steps for both CFG
halves, so it is the most expensive operation in denoising, and a faster attention kernel
directly shortens the time to generate the video.

## 2. The two kernels

Both compute the same attention for the same tokens: each SP rank attends its 2,072 query
tokens to all 4,144 key/value tokens, for 10 heads of 128 dims, for each of the 2 CFG halves.
They differ only in how that work is mapped onto the NeuronCore engines.

- **Legacy: `wan_flash_self_attn`** (`kernels/self_attention.py`, called from
  `models/sp_attention.py`). A hand-written NKI flash attention. One call per CFG half
  (10 heads), heads processed one after another, K/V padded from 4,144 to 6,144 tokens
  (2048-token sections), online softmax, the probability matrix P transposed by multiplying
  it with an identity matrix on the Tensor engine, and every elementwise step on the Vector engine.
- **cte: `attention_cte`** (nkilib, shipped in the image; the kernel staging
  vllm-omni-neuron runs at CP=1, called from `wan/modules/attention.py` with `ATTN_IMPL=cte`).
  One call for both CFG halves (20 head instances), no padding, exp on the Scalar engine with
  the max-subtraction and row-sum fused in, P transposed by DMA, and QK of one query group
  overlapped with PV of an earlier group.

How they fit section 1: the attention math is the same; cte keeps more engines busy at once
and does less wasted work, so each head takes ~0.17 ms instead of ~1.37 ms.

| | legacy | cte |
|---|---|---|
| Runtime per call | 13.686 ms | 1.6735 ms |
| Calls per block per step | 2 (one per CFG half) | 1 (both halves) |
| Hardware / model FLOPS utilization | 9.54% | 33.40% |
| Matmul arithmetic intensity | 2098.75 | 1381.33 |
| HBM read / write per call | 82.69 / 10.63 MiB | 50.59 / 10.12 MiB |
| Self-attention per video, estimated (per-call time x calls) | ~11 s | ~0.7 s |
| Denoise (median of 3 clean runs) | 35.5 s | 25.2 s |
| Total (encode + denoise + decode) | 40.7 s | 30.6 s (-25%) |

Engine busy percentages, read approximately off the Neuron Explorer bar charts
(Summary view, "FLOPS Utilization by Engine" and "DMA Utilization"):

| Engine | legacy | cte |
|---|---|---|
| Tensor, useful (Model %) | ~10% | ~33% |
| Tensor, busy (Active %) | ~28% | ~43% (+~12% active-throttled) |
| Vector | ~88% | ~61% |
| Scalar | ~19% | ~58% |
| GpSimd | ~5% | ~22% |
| DMA queues | ~1-2% | ~60-65% |

## 3. Each change, where it shows in the Analysis view, and where it is in the code

**Per-head structure.** Legacy Tensor/Vector rows repeat in equal ~1.35 ms segments with
short gaps: 10 heads x 1.37 ms = 13.7 ms, the head loop
`for batch_id in nl.sequential_range(batch_size)` (`self_attention.py:65`). cte shows
~0.17 ms blocks, ~10 per core across 1.67 ms (20 head instances).

**1. DMA engines: more activity.**
View: DMA-E64...E92 rows, legacy mostly idle with short bursts, cte all 32 queues busy
almost the whole time; Pending DMA Count, legacy low and flat, cte continuously non-zero.
Code: legacy loads K, V, mask, Q in bursts at each K section inside two sequential loops
(`self_attention.py:65, 81, 87-109`). cte transposes P with DMA
(`attention_cte.py:3412, 3434`, `nisa.dma_transpose`) where legacy uses the Tensor engine
(`self_attention.py:163`), and streams loads alongside compute.

**2. Tensor engine: more useful work.**
View: Tensor/TensorMatrix rows, ~1.35 ms per head (legacy) vs ~0.17 ms (cte); Estimated
Instant MFU rows, legacy low.
Code: in legacy a third of the Tensor-engine work is the identity-matmul transpose
(line 163) and K is padded 4,144 -> 6,144, neither of which is attention math. cte has no
identity transpose and no padding, and overlaps QK for query group i+2 with PV for group i
(`_fused_qkmax_and_pv_impl`, `attention_cte.py:3487`).

**3. Scalar engine: more.**
View: Scalar rows, dense in every cte block.
Code: cte computes exp on the Scalar engine with the max-subtraction and the row-sum fused
into one instruction (`nisa.activation(op=nl.exp, bias=running_max, reduce_op=add)`,
`attention_cte.py:3380-3392`). Legacy does these as separate passes.

**4. GpSimd: more.**
View: GpSimd rows, legacy sparse blips aligned with the DMA bursts, cte dense in every block.
Code: one visible cause is cte initialising its max buffers on GpSimd
(`nisa.memset(..., engine=nisa.gpsimd_engine)`, `attention_cte.py:3509`). The rest of its
GpSimd work is not traced here.

**5. Vector engine: less.**
View: Vector rows, legacy solid across the whole window (~88%), cte interleaved with the
other engines.
Code: legacy runs every per-tile step on the Vector engine (`self_attention.py:117-176`):
PSUM-to-SBUF copy, scale multiply, mask add, max, two f32-to-bf16 copies, and an SBUF add for
the PV running sum. cte moved the exp and its sum to Scalar and the transpose to DMA.

**6. HBM reads and writes: less.**
View: HBM Throughput row and the Summary donut, read 82.69 -> 50.59 MiB, write 10.63 -> 10.12 MiB.
cte reads each input exactly once: 20 head instances x (2,072 Q + 4,144 K + 4,144 V tokens)
x 128 x 2 bytes = 50.6 MiB, and writes the output exactly once: 20 x 2,072 x 128 x 2 bytes
= 10.12 MiB. Legacy reads more because K/V are padded to 6,144 tokens, Q is reloaded for every
K section (line 109 sits inside the section loop at line 81), and the mask is reloaded per
section per head (line 102). Those do not fully add up to 82.69 MiB, and legacy's 10.63 MiB
write is about twice its output; the remainder is not explained by the code.

**Not compared.** cte shows a Throttling row (square wave), matching "Active Throttled" in its
Summary. The legacy screenshot has no Throttling row, so throttling is not compared.

## Files

- `index.html` - illustrated, narrated version of this page (open in a browser).
- `audio/` - narration per section (`say -v Samantha`, AAC).
- `img/` - the six Neuron Explorer screenshots this analysis reads from.
- Kernel source: `kernels/self_attention.py` (legacy); `attention_cte.py` is in the image's
  nkilib and in `dit_pairs_cte/code/` from the capture run.
