# Wan2.2-I2V-A14B on AWS Trainium2

Image-to-video generation with the [Wan-AI/Wan2.2-I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B) model running on AWS Trainium2 with NKI kernels.

## Architecture

- **Model**: 14B parameter dual-model DiT (high_noise + low_noise, boundary at t=0.9)
- **DiT**: dim=5120, 40 heads, 40 layers, ffn_dim=13824
- **VAE**: Wan2.1 VAE, stride (4, 8, 8), 16 latent channels
- **TP=4 or 8**: DiT sharded across the ranks (10 or 5 heads/rank), both experts resident
- **VAE**: fullgraph, decode tiled over all ranks
- **NKI kernels**: nkilib attention_cte, qkv, output projection, conv3d

## Resource Requirements

Both experts (2 x 28.6 GB in bf16), the UMT5 encoder and the VAE fit on **one** Neuron device
(trn2: 96 GiB HBM, 8 physical cores). Measured on trn2.48xlarge at 368x480, 21 frames, 10 steps,
batched CFG, the job's default flags; times are the mean of warm runs 3 and 4:

| Config | Resource claim | Ranks | Device memory per rank (peak) | Total | Denoise | VAE encode / decode |
|---|---|---|---|---|---|---|
| one device, LNC2 (job default) | `s-lnc2` | TP 4 | 20.8 GiB of 24 (rank 0) | **8.6 s** (9.8 s before `+mlp+vstream+qsplit`) | 7.8 s | 0.41 / 0.39 s |
| one device, LNC1 | `s-lnc1` | TP 8 | 12.7 GiB (rank 0); ranks 0+1 share a 24 GiB stack: 21.8 | **10.5 s** | 9.2 s | 0.64 / 0.64 s |
| two devices, LNC2 | `m-lnc2` | TP 8 | -- | **6.35 s** | 5.36 s | 0.42 / 0.53 s |

- `s-lnc2` needs only `TP_DEGREE=4` (and `VAE_TILE_GRID=2x2`, the default for 4 ranks).
- `s-lnc1` also needs `NEURON_LOGICAL_NC_CONFIG=1 TP_DEGREE=8 VAE_ENCODE_RANK0=1 FAST_EMBED_HOST=1`.
  Two cores share each 24 GiB HBM stack, and without the two flags step 0 runs out of memory:
  `VAE_ENCODE_RANK0` encodes the image on rank 0 only and broadcasts `y`; `FAST_EMBED_HOST` keeps
  the text/time embedding weights (0.43 GiB per expert per rank) on the host and caches each
  step's time modulation. Both give bit-identical output (checked at LNC2).
- Step-0 noise prediction vs an fp32 CPU DiT (rel-RMS): `s-lnc2` 0.0362 (0.0389 with
  `+mlp+vstream+qsplit`), `s-lnc1` 0.0376, `m-lnc2` 0.0395.
- The job's default stage is `fast_cte+oproj+adaln+vaetile+t5+qkvnki+xqnki+mlp+vstream+qsplit`
  (`FAST_MLP=1`, `VAE_STREAM=1`, `FAST_QKV_SPLIT=1`; see RESULTS.md). The `s-lnc1` and `m-lnc2`
  rows were measured with the previous stage (`+upnki`, no `+vstream`, no `+qsplit`).

## Deploy

```bash
kubectl apply -f wan2-i2v-14b-job.yaml
kubectl logs -f job/wan2-i2v-14b
```

## What to Observe

### Model Loading Phase (~3 min)
```
Loading T5...
T5 encoded and freed              # T5 freed after prompt encoding
Loading VAE with TP=2...
VAE TP=2 sharded on all ranks
VAE encode done: y shape = [20, 5, 30, 56]
Loading high_noise_model...
[TP] Sharded model on rank 0/8: 5 heads/rank, 1.99B params (local)
Compiling high_noise_model...      # torch.compile on patch_embed, text_embed, head, FFNs
Loading low_noise_model...         # stays on CPU until boundary
DiT ready (high_noise on device, low_noise on CPU)
```

### Denoising Phase
```
Run 1/1 (10 steps)
Step 0: ~170s (first step compiles NEFFs)
Step 1-3: ~4-8s (warm, high_noise_model)
Step 4: swapping to low_noise_model    # boundary crossing at t=0.9
Step 5: ~30s (first step with low_noise compiles its NEFFs)
Step 6-9: ~7-8s (warm, low_noise_model)
```

### Key Metrics
- **First step**: ~170s (NEFF compilation, one-time cost)
- **Warm step**: ~7-8s (actual Neuron execution)
- **Model swap**: ~30s (CPU→device transfer + first compilation)
- **VAE decode**: ~108s
- **Total (cold)**: ~5-6 min

### Output
- Video: `/tmp/wan2_i2v_output.mp4` (17 frames at 16fps)
- Archived to: `/var/mdl/wan2_2_i2v/runs/<timestamp>/wan2_i2v_output.mp4`

## Memory Layout (one device)

Rank 0, allocated tensors (`MEM_LOG=1`), plus the runtime's share from neuron-monitor:

| Component | `s-lnc2` (TP 4) | `s-lnc1` (TP 8) |
|---|---|---|
| VAE (weights + packed NKI conv filters) | 0.50 GiB | 0.50 GiB |
| high + low noise experts (with `FAST_EMBED_HOST`) | 14.15 GiB | 7.05 GiB |
| peak during a run (activations; VAE encode on rank 0) | 17.56 GiB | 10.45 GiB |
| shared scratchpad + NEFF code + constants | 2.0 GiB | 1.4 GiB |

T5 runs on every rank for the prompt and is freed afterwards (`release_device_cache` returns the
cached blocks). Kernel weight layouts are built on the host before the experts move to the
device, so no device-side transposition copies stay in the allocator cache.

## Configuration

Key environment variables in the job manifest:

| Variable | Default | Description |
|----------|---------|-------------|
| `TP_DEGREE` | 8 | DiT tensor parallelism degree |
| `VAE_TP_DEGREE` | 2 | VAE decoder tensor parallelism |
| `NUM_STEPS` | 10 | Denoising steps |
| `FRAME_NUM` | 17 | Output frames (must be 4n+1) |
| `NUM_RUNS` | 1 | Inference iterations (2 for warm benchmark) |
| `USE_NKI_KERNELS` | 1 | Enable NKI attention kernels |

## DLC Image

Uses the latest Neuron DLC with new NKI APIs (`import nki` top-level):
```
421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b:latest
```

## Scaling Up

Two devices (`m-lnc2`, `TP_DEGREE=8`, `VAE_TILE_GRID=2x4`, `NUM_NEURON_DEVICES=2`) give 6.35 s at
the same spec. Resolution and frame count are set by `OUT_H`, `OUT_W`, `FRAME_NUM` (4n+1).
