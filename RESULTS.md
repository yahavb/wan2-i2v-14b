# Wan2.2-I2V-A14B on Trainium2: results log

Spec for every row: 368x480, 21 frames, 10 steps, batched CFG, trn2, 2 Neuron devices
(`m-lnc2`, 8 logical cores at LNC2). Numbers are medians of clean runs (no profiler),
taken from the job's `RESULT` lines. Stages in one table row group ran on the same node
in the same job. Videos were checked by eye (the accuracy gate).

## End-to-end progression

| Stage | Commit | Denoise | s/step | VAE encode | VAE decode + output | Text encode | Total | Video |
|---|---|---|---|---|---|---|---|---|
| legacy (eager DiT, TP4 SP2, own NKI attention) | `a9fc87c` | 35.5 s | 3.55 | 0.73 s | 1.7 s | n/a | **40.7 s** | ok |
| cte (staging `attention_cte`, eager DiT) | `a9fc87c` | 25.2 s | 2.52 | 0.73 s | 1.8 s | n/a | **30.6 s** | ok |
| constmax (staging const-max, eager DiT) | `a9fc87c` | 24.2 s | 2.42 | 0.73 s | 1.7 s | n/a | **29.4 s** | not checked |
| fast_cte (fullgraph DiT, TP8 SP1) | `79b610b` | 6.3 s | 0.63 | 0.72 s | 1.3 s | n/a | **8.3 s** | ok |
| fast_constmax | `cf82511` | 6.6 s | 0.66 | 0.71 s | 1.3 s | n/a | 8.7 s | ok |
| fast_cte + mlp | `bd5358e` | 6.8 s | 0.68 | 0.71 s | 1.3 s | n/a | 8.9 s | — |
| fast_cte + mlp + oproj | `bd5358e` | 6.5 s | 0.65 | 0.72 s | 1.4 s | n/a | 8.6 s | — |
| fast_cte + oproj | `444c754` | 6.1 s | 0.61 | 0.71 s | 1.3 s | n/a | 8.1 s | — |
| fast_cte + oproj + adaln | `444c754` | 5.8 s | 0.58 | 0.72 s | 1.4 s | n/a | **7.9 s** | "a bit funky" (old VAE) |
| + fullgraph VAE (`VAE_IMPL=static`) | `4e7b97f` | 5.8 s | 0.58 | 0.41 s | 0.9 s | 0.611 s | **7.2 s** | good |
| + fullgraph UMT5 (`T5_IMPL=fast`) | `5d71c2d` | 5.8 s | 0.58 | 0.41 s | 0.9 s | 0.035 s | **7.2 s** | good |
| + 8-rank tiled VAE decode (`VAE_IMPL=static_tiled`, 2x4, overlap 4) | `5b4ee20` | 5.8 s | 0.58 | 0.41 s | 0.5 s | 0.035 s | **6.7 s** | good (default) |

Total = VAE encode + denoise + VAE decode + mp4 write. Text encode is measured separately
(second, warm encode of both prompts) and is not in the total.

## DiT block (one fullgraph NEFF = one transformer block, 400 executions per video)

| Stage | Block time | Block MFU | Change |
|---|---|---|---|
| fast_cte | 15.00 ms | 30.4% | baseline |
| fast_constmax | 15.78 ms | 29.1% | +0.84 ms (const-max loses at 4,140 tokens) |
| + mlp (nkilib `mlp`) | 16.28 ms | 28.1% | +1.32 ms (compiler matmuls win at I=1728) |
| + oproj (nkilib `output_projection_cte`) | 14.33 ms | 31.8% | −0.67 ms |
| + oproj + adaln (staging fused adaln) | 13.62 ms | 33.5% | −0.71 ms more |

### Block profile (`fast_cte+oproj+adaln`, one block NEFF, `5b4ee20`)

| Counter | Value | Share of the 13.77 ms block |
|---|---|---|
| Collectives (15 ops) | 5.17 ms | 38% |
| DMA active | 10.90 ms | 79% |
| Tensor engine active | 7.42 ms | 54% |
| Vector engine active | 8.48 ms | 62% |
| Scalar engine active | 4.14 ms | 30% |
| GpSimd active | 1.11 ms | 8% |
| SBUF spill save / reload | 0.59 / 1.35 GB | — |
| HBM read / write | 2.05 / 1.03 GB | — |
| Transpose FLOPs | 93.5 of 811 GFLOP | 11.5% |
| MFU / MBU | 33.2% / 31.2% | — |

Engines overlap, so shares do not add up to 100%. Largest lever: the three per-block
all-reduces of a [2, 4140, 5120] bf16 tensor (85 MB) across 8 ranks on 2 chips
(self-attn O-proj, cross-attn O-proj, FFN down). Candidate fix: TP4 x CFG2 (one CFG half
per chip, all-reduces half the size and intra-chip).

## Attention kernel alone (eager DiT, per call)

| Kernel | Time per call | Calls per block per step | MFU | HBM read / write |
|---|---|---|---|---|
| legacy `wan_flash_self_attn` | 13.686 ms | 2 | 9.54% | 82.69 / 10.63 MiB |
| `attention_cte` | 1.6735 ms | 1 | 33.40% | 50.59 / 10.12 MiB |

Explained in `docs/attention/` (write-up, narrated page, screenshots).

## Findings

- The old channel-parallel VAE (`models/vae_tp.py`) computed the decoder attention and
  RMS norms wrong (contiguous q|k|v slicing, local-channel norm); its video looked "a bit
  funky". The fullgraph VAE (`models/vae_static21.py`) is correct and faster. Do not use
  `VAE_IMPL=tp`.
- Dead ends at this shape: staging const-max self-attention, nkilib `mlp`.
- Costs: VAE warm-up compile ~271 s; the first VAE call in a fresh process ~52 s even with
  a warm cache, and the second can still be slow (2.4-2.9 s seen). The job runs 4 clean
  runs and takes the median of the last 3.
- Tiled VAE decode (8 ranks, fixed 25x18 latent tiles, cross-faded overlaps) shows no
  seams by eye.

## Components and where they live

| Component | Implementation | Switch |
|---|---|---|
| DiT | `models/fast_dit.py` (fullgraph per block, TP8, in-graph all-reduce, both experts resident) | `FAST_DIT=1` |
| Attention | nkilib `attention_cte` (self + cross) | `FAST_ATTN_IMPL=cte` |
| O-proj | nkilib `output_projection_cte` | `FAST_OPROJ=1` |
| AdaLN | staging `adaln_quant` (vendored, `kernels/nkilib/experimental/norm/`) | `FAST_ADALN=1` |
| VAE | `models/vae_static21.py` (static cache, nkilib conv3d + attention_cte in-graph; decode tiled over 8 ranks) | `VAE_IMPL=static_tiled` |
| Text encoder | `models/fast_t5.py` (UMT5 TP8, fullgraph per block) | `T5_IMPL=fast` |

## Next

- torch.compile scope ladder: `docs/torch_compile_scope.md` (branch `exp/compile-scope`).
- TP4 x CFG2 to cut the cross-chip all-reduce cost.
