# Wan2.2-I2V-A14B on AWS Trainium2

Image-to-video generation with the [Wan-AI/Wan2.2-I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B) model running on AWS Trainium2 with NKI kernels.

## Architecture

- **Model**: 14B parameter dual-model DiT (high_noise + low_noise, boundary at t=0.9)
- **DiT**: dim=5120, 40 heads, 40 layers, ffn_dim=13824
- **VAE**: Wan2.1 VAE, stride (4, 8, 8), 16 latent channels
- **TP=8**: DiT sharded across 8 NeuronCores (5 heads/rank, ~2B params/rank)
- **VAE TP=2**: Decoder channel-parallel across 2 ranks
- **NKI kernels**: Flash self-attention, cross-attention, RoPE (new `import nki` API)

## Resource Requirements

| Config | Resource Claim | NeuronCores | HBM per NC |
|--------|---------------|-------------|------------|
| Single ND | `s-lnc1-trn2` | 8 | ~4GB |
| Dual ND | `m-trn2` | 8 (LNC2) | ~6GB |

The single ND config (`s-lnc1-trn2`) requires:
- Resolution: 256x448, 17 frames (seq_len=2,100)
- Single DiT model on device at a time (swap at boundary)
- VAE TP=2 (TP=8 triggers compiler instruction limit)

The dual ND config (`m-trn2`) supports:
- Resolution: 480x832, 33 frames (seq_len=13,572)
- Both DiT models resident simultaneously

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

## Memory Layout (s-lnc1-trn2)

Each of 4 HBMs (16GB) is shared by 2 NeuronCores:

| Component | Per-rank HBM |
|-----------|-------------|
| DiT weights (1 model, TP=8) | ~2GB |
| Compiled NEFFs (code) | ~140MB |
| Scratchpad | 512MB |
| Activations (seq_len=2100) | ~60MB |
| VAE (ranks 0,1 only) | ~150MB |
| Collectives + runtime | ~60MB |
| **Total per NC** | **~3GB** |

T5 is freed after encoding. Only one DiT model on device at a time. VAE decoder sharded across 2 ranks.

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

To run at higher resolution (480x832, 33+ frames), use `m-trn2` resource claim which provides 2 NDs with more HBM headroom. Edit the job manifest:
- Change `s-lnc1-trn2` → `m-trn2`
- Set `FRAME_NUM=33`
- Update `max_area` in `inference_neuron_i2v.py` to `480 * 832`
- Can load both DiT models simultaneously (no swap needed)
