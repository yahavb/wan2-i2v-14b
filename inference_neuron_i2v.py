"""Wan2.2-I2V-A14B inference with TP8 on Neuron (m-trn2: 2 NDs, LNC2, 8 NeuronCores).

Uses torchrun --nproc_per_node=8 for tensor parallelism.
All models loaded directly on Neuron — no CPU offloading.
2 NDs × 4 NCs (LNC2) = 8 NeuronCores, each with ~6GB HBM.

Architecture (Wan2.2-I2V-A14B):
  dim=5120, 40 heads, 40 layers, ffn_dim=13824
  TP=8: 5 heads/rank, ~1.99B params/rank (~4GB bf16)
  Dual model: low_noise_model (t < boundary), high_noise_model (t >= boundary)
  VAE stride: (4, 8, 8), in_dim=16
"""
import os
import sys
import time
import math
import random
import logging
import gc

import torch
import torch.distributed as dist
import numpy as np
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [rank %(process)d]: %(message)s',
    stream=sys.stdout,
    force=True,
)
for name in ['torch', 'transformers', 'torch_neuronx', 'torch_mlir']:
    logging.getLogger(name).setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

WAN_DIR = os.environ.get("WAN_DIR", os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WAN_DIR)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

MODEL_PATH = os.environ.get("MODEL_PATH", "/tmp/Wan2.2-I2V-A14B")
TP_DEGREE = int(os.environ.get("TP_DEGREE", "4"))
SP_DEGREE = int(os.environ.get("SP_DEGREE", "2"))
T5_RANK = int(os.environ.get("T5_RANK", "0"))
VAE_RANK = 0

NEURON_DEVICE = torch.device("neuron")


def setup_distributed():
    dist.init_process_group(backend="neuron")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.neuron.set_device(local_rank)
    return dist.get_rank(), dist.get_world_size()


FRAME_NUM = int(os.environ.get("FRAME_NUM", "81"))
INSTANCE_TYPE = os.environ.get("INSTANCE_TYPE", "trn2.48xlarge")
LNC_CONFIG = os.environ.get("LNC_CONFIG", "LNC2")
NUM_NEURON_DEVICES = os.environ.get("NUM_NEURON_DEVICES", "1")
USE_NKI_KERNELS = os.environ.get("USE_NKI_KERNELS", "1")


def main():
    rank, world_size = setup_distributed()
    torch.set_grad_enabled(False)

    if rank == 0:
        logger.info("=" * 70)
        logger.info("  Wan2.2-I2V-A14B  TP4 on Neuron")
        logger.info("=" * 70)
        logger.info(f"  Instance:        {INSTANCE_TYPE}")
        logger.info(f"  LNC config:      {LNC_CONFIG}")
        logger.info(f"  NeuronDevices:   {NUM_NEURON_DEVICES}")
        logger.info(f"  TP degree:       {TP_DEGREE}")
        logger.info(f"  SP degree:       {SP_DEGREE}")
        logger.info(f"  World size:      {world_size}")
        logger.info(f"  NKI kernels:     {USE_NKI_KERNELS}")
        logger.info(f"  T5 rank:         {T5_RANK}")
        logger.info(f"  Frame count:     {FRAME_NUM}")
        logger.info("=" * 70)

    from wan.configs import WAN_CONFIGS
    from wan.modules.model import WanModel
    from wan.modules.t5 import T5EncoderModel
    from wan.modules.vae2_1 import Wan2_1_VAE
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
    from models.tp_utils import init_tp_group, shard_model_tp, get_tp_rank
    from models.vae_tp import create_vae_tp_group, shard_vae_model_tp
    from models.parallel_state import init_sp_group
    from models.sp_attention import patch_model_for_sp
    from models.sp_model import make_sp_forward

    config = WAN_CONFIGS['i2v-A14B']

    # ── Encode prompt with T5 (then free it — not needed during denoising) ──
    prompt = (
        "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. "
        "The fluffy-furred feline gazes directly at the camera with a relaxed expression. "
        "Blurred beach scenery forms the background featuring crystal-clear waters, "
        "distant green hills, and a blue sky dotted with white clouds."
    )
    n_prompt = config.sample_neg_prompt

    ctx_tensor = torch.zeros(1, 512, 4096, dtype=torch.bfloat16, device=NEURON_DEVICE)
    ctx_null_tensor = torch.zeros(1, 512, 4096, dtype=torch.bfloat16, device=NEURON_DEVICE)

    if rank == T5_RANK:
        logger.info("Loading T5...")
        text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(MODEL_PATH, config.t5_checkpoint),
            tokenizer_path=os.path.join(MODEL_PATH, config.t5_tokenizer))
        text_encoder.model = text_encoder.model.to(NEURON_DEVICE)
        context = text_encoder([prompt], NEURON_DEVICE)
        context_null = text_encoder([n_prompt], NEURON_DEVICE)
        ctx_tensor[0, :context[0].shape[0]] = context[0].to(torch.bfloat16)
        ctx_null_tensor[0, :context_null[0].shape[0]] = context_null[0].to(torch.bfloat16)
        del text_encoder, context, context_null
        gc.collect()
        logger.info("T5 encoded and freed")

    dist.broadcast(ctx_tensor, src=T5_RANK)
    dist.broadcast(ctx_null_tensor, src=T5_RANK)

    if rank == 0:
        logger.info("Prompt encoded and broadcast")

    # ── VAE encode image (before DiT loading — needs scratch space for Conv3D) ──
    import torchvision.transforms.functional as TF

    image_path = os.path.join(MODEL_PATH, "examples/i2v_input.JPG")
    if not os.path.exists(image_path):
        image_path = os.path.join(MODEL_PATH, "examples/i2v_input.jpg")
    if not os.path.exists(image_path):
        if rank == 0:
            logger.info("No example image found, using random image")
        img = Image.new("RGB", (1280, 720), (128, 128, 128))
    else:
        img = Image.open(image_path).convert("RGB")

    vae_stride = config.vae_stride  # (4, 8, 8)
    patch_size = config.patch_size  # (1, 2, 2)
    frame_num = FRAME_NUM
    max_area = 832 * 480  # 480p output (was 320*576). Bigger image; enables VAE W-shard.

    ih, iw = img.height, img.width
    aspect_ratio = ih / iw
    lat_h = round(
        np.sqrt(max_area * aspect_ratio) // vae_stride[1] // patch_size[1] * patch_size[1])
    lat_w = round(
        np.sqrt(max_area / aspect_ratio) // vae_stride[2] // patch_size[2] * patch_size[2])
    # round lat_w UP to a multiple of world_size so VAE W-shard engages for any
    # input aspect ratio (the model pads seq_len anyway). world = TP*SP = all ranks.
    _wsh = world_size
    if lat_w % _wsh != 0:
        lat_w += _wsh - (lat_w % _wsh)
    oh = lat_h * vae_stride[1]
    ow = lat_w * vae_stride[2]

    scale = max(ow / iw, oh / ih)
    img = img.resize((round(iw * scale), round(ih * scale)), Image.LANCZOS)
    x1 = (img.width - ow) // 2
    y1 = (img.height - oh) // 2
    img = img.crop((x1, y1, x1 + ow, y1 + oh))

    img_tensor = TF.to_tensor(img).sub_(0.5).div_(0.5)

    F = frame_num
    T_latent = (F - 1) // vae_stride[0] + 1
    seq_len = T_latent * lat_h * lat_w // (patch_size[1] * patch_size[2])

    seed = 42
    seed_g = torch.Generator(device=torch.device("cpu"))
    seed_g.manual_seed(seed)

    z_dim = 16
    noise = torch.randn(
        z_dim, T_latent, lat_h, lat_w,
        dtype=torch.float32, generator=seed_g, device=torch.device("cpu"))

    # Build I2V mask
    msk = torch.ones(1, F, lat_h, lat_w)
    msk[:, 1:] = 0
    msk = torch.concat([
        torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
    ], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
    msk = msk.transpose(1, 2)[0]

    VAE_TP_DEGREE = int(os.environ.get("VAE_TP_DEGREE", "8"))
    VAE_TP_RANKS = list(range(VAE_TP_DEGREE))

    y_device = torch.zeros((z_dim + 4, T_latent, lat_h, lat_w), dtype=torch.bfloat16, device=NEURON_DEVICE)

    # Load VAE on all VAE TP ranks, shard decoder
    if rank in VAE_TP_RANKS:
        if rank == 0:
            logger.info(f"Loading VAE with TP={VAE_TP_DEGREE}...")
        vae = Wan2_1_VAE(
            vae_pth=os.path.join(MODEL_PATH, config.vae_checkpoint),
            device=torch.device('cpu'))
        vae.model = vae.model.to(device=NEURON_DEVICE, dtype=torch.bfloat16)
        vae.mean = vae.mean.to(device=NEURON_DEVICE, dtype=torch.bfloat16)
        vae.std = vae.std.to(device=NEURON_DEVICE, dtype=torch.bfloat16)
        vae.scale = [vae.mean, 1.0 / vae.std]

        # VAE WIDTH-SHARD (ported from rolling_forcing): full conv weights on
        # every rank, sharded along spatial WIDTH with halo exchange. No channel
        # TP. init the W-group so CausalConv3d/AttentionBlock shard internally.
        from models.parallel_state import init_vae_w_group
        init_vae_w_group()
        if rank == 0:
            logger.info(f"VAE WIDTH-SHARD across {VAE_TP_DEGREE} ranks (halo exchange)")

        # Encode image (encoder is replicated, all ranks get same result)
        img_for_vae = torch.nn.functional.interpolate(
            img_tensor[None], size=(oh, ow), mode='bicubic').transpose(0, 1)
        img_seq = torch.cat([img_for_vae, torch.zeros(3, F - 1, oh, ow)], dim=1)
        z = vae.encode([img_seq.to(device=NEURON_DEVICE, dtype=torch.bfloat16)])
        z_result = z[0].to(torch.bfloat16)
        msk_device = msk.to(device=NEURON_DEVICE, dtype=torch.bfloat16)
        y_device = torch.cat([msk_device, z_result], dim=0)
        if rank == 0:
            logger.info(f"VAE encode done: y shape = {y_device.shape}")
        del z, z_result, img_seq, img_for_vae
        gc.collect()
    else:
        vae = None

    # Broadcast y to any ranks not in VAE TP group (if VAE_TP_DEGREE < TP_DEGREE)
    if VAE_TP_DEGREE < TP_DEGREE:
        dist.broadcast(y_device, src=VAE_RANK)

    # ── Load DiT models with TP + SP sharding ──
    init_tp_group(tp_degree=TP_DEGREE)
    init_sp_group(tp_degree=TP_DEGREE)
    tp_rank = get_tp_rank()

    if rank == 0:
        logger.info(f"Loading high_noise_model (on device first — runs for t >= boundary)...")
    high_noise_model = WanModel.from_pretrained(MODEL_PATH, subfolder=config.high_noise_checkpoint)
    high_noise_model.eval().requires_grad_(False)
    shard_model_tp(high_noise_model, tp_rank, TP_DEGREE)
    patch_model_for_sp(high_noise_model)
    make_sp_forward(high_noise_model)
    high_noise_model = high_noise_model.to(torch.bfloat16).to(NEURON_DEVICE)

    if rank == 0:
        logger.info("Compiling high_noise_model...")
    high_noise_model.patch_embedding = torch.compile(high_noise_model.patch_embedding, backend='neuron', dynamic=False)
    high_noise_model.text_embedding = torch.compile(high_noise_model.text_embedding, backend='neuron', dynamic=False)
    high_noise_model.head = torch.compile(high_noise_model.head, backend='neuron', dynamic=False)
    for block in high_noise_model.blocks:
        block.ffn = torch.compile(block.ffn, backend='neuron', dynamic=False)

    if rank == 0:
        logger.info(f"Loading low_noise_model (stays on CPU until boundary)...")
    low_noise_model = WanModel.from_pretrained(MODEL_PATH, subfolder=config.low_noise_checkpoint)
    low_noise_model.eval().requires_grad_(False)
    shard_model_tp(low_noise_model, tp_rank, TP_DEGREE)
    patch_model_for_sp(low_noise_model)
    make_sp_forward(low_noise_model)
    low_noise_model = low_noise_model.to(torch.bfloat16)

    dist.barrier()
    if rank == 0:
        logger.info("DiT ready (high_noise on device, low_noise on CPU)")

    if rank == 0:
        logger.info("-" * 70)
        logger.info(f"  Image: {oh}x{ow}, Frames: {frame_num}, Seq len: {seq_len}")
        logger.info(f"  Latent: [{z_dim}, {T_latent}, {lat_h}, {lat_w}]")
        logger.info(f"  DiT: dim={config.dim}, heads={config.num_heads}, layers={config.num_layers}")
        logger.info(f"  Heads/rank: {config.num_heads // TP_DEGREE}, Boundary: {config.boundary}")
        logger.info("-" * 70)

    noise = noise.to(torch.bfloat16).to(NEURON_DEVICE)

    NUM_STEPS = int(os.environ.get("NUM_STEPS", "10"))
    NUM_RUNS = int(os.environ.get("NUM_RUNS", "1"))

    arg_c = {'context': [ctx_tensor[0]], 'seq_len': seq_len, 'y': [y_device]}
    arg_null = {'context': [ctx_null_tensor[0]], 'seq_len': seq_len, 'y': [y_device]}

    latent_orig = noise.clone()
    run_results = []
    boundary = config.boundary * config.num_train_timesteps
    current_on_device = 'high'

    for run_idx in range(NUM_RUNS):
        run_label = f"Run {run_idx+1}/{NUM_RUNS}"
        if rank == 0:
            logger.info(f"═══ {run_label} ({NUM_STEPS} steps) ═══")

        latent = latent_orig.clone()

        sample_scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=config.num_train_timesteps,
            shift=1, use_dynamic_shifting=False)
        sample_scheduler.set_timesteps(NUM_STEPS, device=torch.device("cpu"), shift=config.sample_shift)
        timesteps = sample_scheduler.timesteps

        run_start = time.time()
        denoise_start = time.time()

        for step_idx, t in enumerate(tqdm(timesteps, disable=(rank != 0), desc=run_label)):
            latent_model_input = [latent]
            timestep = torch.tensor([t.item()], device=NEURON_DEVICE)

            if t.item() >= boundary:
                needed = 'high'
                guide_scale = config.sample_guide_scale[1]
            else:
                needed = 'low'
                guide_scale = config.sample_guide_scale[0]

            if needed != current_on_device:
                if rank == 0:
                    logger.info(f"  Step {step_idx}: swapping to {needed}_noise_model")
                if current_on_device == 'high':
                    high_noise_model = high_noise_model.to('cpu')
                    low_noise_model = low_noise_model.to(NEURON_DEVICE)
                else:
                    low_noise_model = low_noise_model.to('cpu')
                    high_noise_model = high_noise_model.to(NEURON_DEVICE)
                current_on_device = needed
                gc.collect()

            model = high_noise_model if current_on_device == 'high' else low_noise_model

            # Batched CFG: cond+uncond in ONE B=2 forward (shared latent/t/y,
            # differ only in context). ~halves denoise vs two sequential forwards.
            out = model(
                [latent, latent], t=timestep, seq_len=seq_len,
                context=[ctx_tensor[0], ctx_null_tensor[0]],
                y=[y_device, y_device])
            noise_pred_cond = out[0]
            noise_pred_uncond = out[1]


            noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)

            temp_x0 = sample_scheduler.step(
                noise_pred.cpu().unsqueeze(0), t,
                latent.cpu().unsqueeze(0),
                return_dict=False, generator=seed_g)[0]
            latent = temp_x0.squeeze(0).to(NEURON_DEVICE)

        denoise_time = time.time() - denoise_start
        if rank == 0:
            logger.info(f"{run_label} denoising: {denoise_time:.1f}s ({denoise_time/NUM_STEPS:.1f}s/step)")

        # ── Free DiT for VAE decode ──
        if current_on_device == 'high':
            high_noise_model = high_noise_model.to('cpu')
        else:
            low_noise_model = low_noise_model.to('cpu')
        gc.collect()

        # ── Decode with VAE (WIDTH-sharded across all ranks + halo exchange) ──
        vae_start = time.time()
        if rank in VAE_TP_RANKS:
            if rank == 0:
                logger.info(f"{run_label} VAE decode (W-shard x{VAE_TP_DEGREE})...")
            from wan.modules.vae2_1 import set_vae_w_active
            from models.parallel_state import get_vae_w_degree, get_vae_w_rank
            lat = latent.to(torch.bfloat16)
            wsh = get_vae_w_degree()
            # Only width-shard if W divides evenly across ranks; else full decode.
            do_wshard = wsh > 1 and (lat.shape[-1] % wsh == 0)
            if wsh > 1 and not do_wshard and rank == 0:
                logger.info(f"VAE W-shard SKIPPED: lat_w={lat.shape[-1]} not divisible by {wsh}; full decode")
            if do_wshard:
                Wl = lat.shape[-1] // wsh
                r = get_vae_w_rank()
                lat = lat[..., r * Wl:(r + 1) * Wl].contiguous()
                set_vae_w_active(True)        # enable halo/gather inside the decoder
            videos = vae.decode([lat])
            set_vae_w_active(False)

            # DEBUG: compare W-shard output vs a full (non-sharded) decode on the
            # SAME latent. All ranks run the full decode (no collectives in it when
            # inactive) so it's safe; only rank 0 compares. Finds if halo math is wrong.
            if os.environ.get("WSHARD_DEBUG", "0") == "1" and do_wshard and run_idx == 0:
                full = vae.decode([latent.to(torch.bfloat16)])   # W-shard inactive -> full decode
                if rank == 0:
                    a = videos[0].float(); bvid = full[0].float()
                    if a.shape == bvid.shape:
                        d = (a - bvid).abs().max().item()
                        logger.info(f"[WSHARD_DEBUG] W-shard vs full max|diff|={d:.4f} "
                                    f"-> {'MATCH' if d < 0.05 else 'DIVERGES'}")
                    else:
                        logger.info(f"[WSHARD_DEBUG] SHAPE MISMATCH wshard={tuple(a.shape)} full={tuple(bvid.shape)}")

            if rank == 0:
                video = videos[0]
                import imageio
                output_path = "/tmp/wan2_i2v_output.mp4"
                video_cpu = video.cpu().float()
                video_np = ((video_cpu.clamp(-1, 1) * 0.5 + 0.5) * 255).byte()
                if video_np.dim() == 4 and video_np.shape[0] == 3:
                    video_np = video_np.permute(1, 2, 3, 0)
                frames = [video_np[i].numpy() for i in range(video_np.shape[0])]
                imageio.mimwrite(output_path, frames, fps=16, codec='libx264')
                logger.info(f"Video saved to {output_path} ({len(frames)} frames)")

        dist.barrier()
        vae_time = time.time() - vae_start
        run_time = time.time() - run_start

        if rank == 0:
            logger.info(f"═══ {run_label} DONE: denoise={denoise_time:.1f}s, vae={vae_time:.1f}s, total={run_time:.1f}s ═══")
            run_results.append({
                'run': run_idx + 1,
                'denoise': denoise_time,
                'vae': vae_time,
                'total': run_time,
            })

    if rank == 0 and run_results:
        logger.info("")
        logger.info("=" * 70)
        logger.info("  BENCHMARK SUMMARY")
        logger.info("=" * 70)
        logger.info(f"  TP={TP_DEGREE}, {config.num_heads//TP_DEGREE} heads/rank")
        logger.info(f"  {oh}x{ow}, {frame_num} frames, {NUM_STEPS} steps")
        for r in run_results:
            logger.info(f"  Run {r['run']}: denoise={r['denoise']:.1f}s ({r['denoise']/NUM_STEPS:.1f}s/step), vae={r['vae']:.1f}s, total={r['total']:.1f}s")
        logger.info("=" * 70)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
