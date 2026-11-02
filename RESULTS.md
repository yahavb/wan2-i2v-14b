# Wan2.2-I2V-A14B on Trainium2: results log

## One Neuron device (branch `loop_engineering`)

Same spec as below (368x480, 21 frames, 10 steps, batched CFG, the validated default flags
`fast_cte+oproj+adaln+vaetile+t5+qkvnki+xqnki+upnki`), trn2.48xlarge, both experts resident.
Times: mean of runs 3 and 4 of 4 in one process.

| Config | Claim | Total | Denoise | VAE encode | VAE decode | step-0 vs fp32 | Video vs 2 devices |
|---|---|---|---|---|---|---|---|
| 2 devices, LNC2, TP 8 (reference, same host) | `m-lnc2` | 6.35 s | 5.36 s | 0.47 s | 0.53 s | 0.0395 | -- |
| 1 device, LNC2, TP 4 | `s-lnc2` | **9.84 s** | 8.71 s | 0.42 s | 0.70 s | 0.0362 | 25.7 dB (worst frame 24.2) |
| 1 device, LNC1, TP 8, `VAE_ENCODE_RANK0=1 FAST_EMBED_HOST=1` | `s-lnc1` | **10.50 s** | 9.23 s | 0.64 s | 0.64 s | 0.0376 | -- |

- step-0 = rel-RMS of the step-0 CFG noise prediction against `scripts/validate/ref_fp32.py`
  (fp32 CPU DiT on the same input). The single-device outputs differ from the 2-device run only
  by accumulation order (TP 4 vs 8), like any kernel change; frames checked by eye.
- `s-lnc2` runs `main`'s code unchanged (`TP_DEGREE=4`, `VAE_TILE_GRID=2x2`). This branch's code
  changes are bit-identical at LNC2 (latent, frames, mp4 and step-0 equal to `main`):
  host-side kernel weight layouts, `release_device_cache` between phases, `VAE_ENCODE_RANK0`,
  `FAST_EMBED_HOST`.
- `s-lnc1` ran out of device memory without the two flags: ranks 2i and 2i+1 share a 24 GiB HBM
  stack, and the experts take 7.05 GiB per rank. First failure: the per-run VAE encode on every
  rank (66 MiB cache tensors); then the DiT activations at step 0 and the decoder NEFF load.
- Device memory peak (neuron-monitor): `s-lnc2` rank 0 20.8 GiB of 24 (17.6 GiB allocated tensors);
  `s-lnc1` rank 0 12.7 GiB, HBM stack of ranks 0+1 21.8 of 24 GiB.

### `s-lnc2` speedups (same spec, runs 2-4 of 4)

| Change | Flags | Block | Denoise | VAE decode + output | Total | step-0 vs fp32 |
|---|---|---|---|---|---|---|
| one device, TP 4 (above) | default stage | 21.51 ms | 8.71 s | 0.70 s | 9.84 s | 0.0362 |
| FFN through nkilib `mlp` (up + GELU + down in one kernel), not the qkv-kernel up projection | `FAST_MLP=1 FAST_FFNUP_NKI=0` (`+mlp`, no `+upnki`) | 20.16 ms | 8.14 s | 0.65 s | 9.20 s | 0.0348 |
| + mp4 encoded during the decode: blend, uint8 and ffmpeg per latent frame; ffmpeg spawned without fork | `+VAE_STREAM=1` (`+vstream`) | 20.16 ms | 8.16 s | 0.46 s | 9.03 s | 0.0348 |
| + self-attention q, k and v as three nkilib qkv calls on [5120, 1280] weights, not one on the fused [5120, 3840] | `+FAST_QKV_SPLIT=1` (`+qsplit`) | 19.36 ms | 7.78 s | 0.39 s | 8.59 s | 0.0389 |
| + residual stream sharded by token between the 4 ranks (Megatron sequence parallelism): reduce-scatter + all-gather instead of each all-reduce, residual adds, gates and norm2 on a quarter of the rows | `+FAST_SP=1` (`+sp`) | 19.05 ms | 7.66 s | 0.39 s | 8.47 s | 0.0348 |
| + norm1 and norm3 (with its gated residual) on the rank's rows too, in the adaln kernel, all-gathered normalized; the q / k / v and cross-q kernels without their fused LayerNorm | `+FAST_SP_LN=1` (`+spln`) | 18.51 ms | 7.44 s | 0.42 s | **8.28 s** | 0.0390 |

- Block = one DiT block NEFF on real inputs (`scripts/validate/block_bench.py`, layer 0 of the high
  expert at step 0, `DUMP_BLOCK_IN`). At TP 8 on two devices nkilib `mlp` lost (+1.32 ms, I = 1728);
  at TP 4 (I = 3456) it wins 1.35 ms per block. rel-RMS of the block output vs fp32 CPU: 0.00344
  (default 0.00346).
- `VAE_STREAM`: same frames and the same mp4 bytes as the unstreamed decode (checked with the
  same DiT flags). Before it, the warm VAE phase was tile decode 0.19 s, gather 0.01 s, blend on
  the host 0.07 s, uint8 0.06 s, mp4 write 0.39 s (`VAE_TIMING=1`).
- Video vs the default-flags single-device run: 24.4 dB (accumulation order), checked by eye.
- `FAST_QKV_SPLIT`: the fused q|k|v weight is 39 MB per rank at TP 4, more than one core's 24 MB
  SBUF; each [5120, 1280] weight is 13 MB. The LayerNorm runs once per call (three times). Block
  output vs the fused kernel's: rel-RMS 1e-5. Block vs fp32 CPU: 0.00344 at layer 0 and 0.00361
  at layer 20, the same as the fused kernel at both layers. The step-0 change (0.0348 -> 0.0389) is
  accumulation order through 40 blocks, still below the 2-device run's 0.0395. Video vs the
  `+vstream` run: 25.1 dB, checked by eye.
- `FAST_SP`: the collectives move the same bytes as the three all-reduces; the gain is the
  replicated elementwise work, now on 2070 of the 8280 rows (the patch embedding keeps the rank's
  rows, the head runs on them and all-gathers its 64-wide output). Its output (latent, frames, mp4)
  is byte-identical to the `+mlp` run's: the split q / k / v and the sequence-parallel sums happen
  to round like the fused kernel with all-reduces. Block bench with `FAST_SP` (rank 0's rows vs the
  unsharded block): rel-RMS 2e-5 (high expert layer 0), 1e-5 (low expert layer 39), 0.27 / 0.30 ms
  faster.
- `FAST_SP_LN`: without sequence parallelism, one shared norm1 before plain q / k / v kernels was no
  faster (19.36 vs 19.31 ms): the adaln kernel on all 8280 rows costs what the three fused
  LayerNorms do. On a quarter of the rows it wins 0.55 ms. Block vs fp32 CPU (rank 0's rows):
  0.00305 (`+sp` 0.00321).
- Runs 2-4 of `+spln`: 8.24, 8.22, 8.37 s (run 4's VAE phase 0.52 s, the others 0.37-0.38 s).
  Video vs the 2-device reference run: `+spln` 27.4 dB (worst frame 25.5), the closest of the
  single-device runs (default stage 25.7 dB, `+mlp` / `+sp` 23.2, `+qsplit` 22.5); checked by eye.
- `+sp` and `+spln` were measured at `s-lnc2` (TP 4) only.
- Block ablations with `FAST_MLP` (ms of the 20.16 ms block): self-attention 3.28 (attention_cte
  and the q / k / v layout copies), all-reduces 1.49, cross-attention 0.69, RoPE 0.52, the whole
  self-attention prologue (QK-norm, RoPE, scale, layout copies) 0.82. Collectives alone: the 3
  all-reduces of a [2, 4140, 5120] bf16 tensor 3.19 ms; as 3 reduce-scatters + all-gathers 2.90 ms.
- No gain (block ms against 20.16-20.20 at the time): K pre-transposed with the softmax scale
  inside attention_cte 20.19; the 5B branch's `attention_cte_lb` kernel 20.22; cross-attention K/V
  stored in the kernel's layout 20.23; one torch q|k|v matmul (`FAST_QKV=1`) instead of the qkv
  kernel 20.12; the FFN kernel over 2 or 4 slices of its hidden dim (weights that fit SBUF, the
  q / k / v split's idea) 20.12 / 20.30; the cross-attention q by matmul (`FAST_CROSSQ_NKI=0`) 21.00.
- No gain on the `+qsplit` block (19.28-19.32 ms):
  - the two CFG halves as independent chains in one graph, so one half's all-reduce could run
    during the other half's kernels: 20.57 ms, the same as running them one after the other
    (20.59). The compiler schedules one half after the other; halving the kernels' rows costs
    1.27 ms. `--tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2'`
    changed nothing here or on the `+sp` block;
  - q and k each normalized and rotated right after its kernel (separate sum-of-squares
    all-reduces): 19.34;
  - RoPE on [real | imaginary] halves (q / k weight columns permuted on the host): 19.28;
  - cross-attention over 128 keys instead of 512 (the 442 / 386 pad tokens of the two prompts
    share one key and value; the missing copies added back exactly from the kernel's softmax
    statistics): 21.58. The kernel is no faster (its time is the q side: 4140 queries x 20 heads)
    and the correction's elementwise ops on the d-major layout cost 2.3 ms.
- Refuted: const-max self-attention (the 5B branch's fused `attention_cmax` kernel, QK-norm + RoPE
  inside, ported to LNC2 with one CFG half per core): 21.51 -> 20.04 ms per block and step-0
  accurate, but the video was NaN. The QK-norm weight bound (sqrt(d) * max|g_q| * max|g_k| per head)
  is not a bound here: on the low-noise expert's first step (t = 882), layer 2 head 23 reaches a
  scaled score of 433 against a weight bound of 31.7 (per-head |q| up to 76.6), so exp(score - 15)
  overflows. A constant max is unsafe for this model's activations.

## Two Neuron devices

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

## trn3 (`trn3-dev1.48xlarge`, 2 devices, LNC2, TP8), branch `latency_loop`

Same spec (368x480, 21 frames, 10 steps, batched CFG). Latency = run 3 and 4 of 4 in one process
(compile and the first loaded runs excluded). Block = the DiT block NEFF (400 executions per
video), profiled across all 8 ranks; engine numbers from `scripts/engine_overlap.py` on its
instruction table. Every row is in `perf_ledger.jsonl`.

| Change | Flag | Block | Tensor busy | 1 engine busy | Total | Frames vs baseline |
|---|---|---|---|---|---|---|
| baseline (trn2 default config) | -- | 10.60 ms | 7.22 ms | 2.63 ms | **5.10 s** | reference |
| one q\|k\|v matmul | `FAST_QKV=1` | 10.71 ms | 7.05 ms | 2.81 ms | 5.1 s | refuted |
| adaln+QKV in 4 torch-level chunks | `FAST_QKV_CHUNKS=4` | 11.29 ms | 7.24 ms | 3.38 ms | -- | refuted |
| norm1+mod+QKV+bias in nkilib `qkv` kernel | `FAST_QKV_NKI=1` | 10.22 ms | 6.71 ms | 2.43 ms | **4.90 s** | 25.8 dB |
| + cross-attn q in the same kernel | `+FAST_CROSSQ_NKI=1` | 9.64 ms | 7.17 ms | 2.10 ms | **4.72 s** | 23.0 dB |
| + norm2+mod+FFN up in the same kernel | `+FAST_FFNUP_NKI=1` | 9.48 ms | 7.23 ms | 2.61 ms | **4.65 s** | 24.8 dB |
| + cross-attn residual in the adaln kernel | `+FAST_XRES_KERNEL=1` | 10.12 ms | 6.74 ms | 2.56 ms | 4.87 s | refuted |

**Validated** (`MODE=validate`, `scripts/validate/validate_job.sh`, `FAST_QKV_NKI=1 FAST_CROSSQ_NKI=1
FAST_FFNUP_NKI=1`), all PASS:

| Check | Candidate | Default (flags off) |
|---|---|---|
| total, run 3, same pod | 4.63 s | 5.14 s |
| step-0 noise pred rel-RMS vs fp32 CPU DiT | 0.0396 | 0.0418 (plain-torch attention: 0.0311) |
| video PSNR vs plain-torch bf16 trajectory | 23.15 dB (worst frame 21.58) | 23.03 dB (worst 21.25) |
| VAE PSNR vs fp32 CPU decode | 43.08 dB | 43.11 dB |
| main == flags-off branch (mp4 bytes) | -- | identical |
| deterministic (candidate twice) | identical | -- |

Frames differ from the baseline for every kernel change (different accumulation order, amplified
over 10 steps); quality is judged by `MODE=validate` (`scripts/validate/`), not by the baseline
PSNR above.

Findings:
- AdaLN with a uniform timestep is LayerNorm with gamma = 1 + scale, beta = shift, so nkilib
  `qkv` (CTE) fuses it with the following matmul and pipelines the vector-engine norm against
  the tensor-engine matmul inside one kernel. Splitting the same work at torch level does not
  overlap: the compiler re-merges the chunk matmuls and every NKI custom call is a barrier.
- With the three fusions the tensor engine is busy 76% of the block (7.23 of 9.48 ms) and is
  the bound. Remaining tensor time: the fused-kernel matmuls 2.5 ms, attention 1.3 ms, FFN down
  1.0 ms, O-proj 0.7 ms, transposes ~0.9 ms (compiler-inserted around the residual adds
  `x + gate * y`, plus the kernel's input transpose at `qkv_cte.py:2192`).

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
