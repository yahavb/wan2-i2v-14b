"""HTTP endpoint for Wan2.2-I2V-A14B on Neuron (m-trn2: 2 NDs, LNC2, 8 NeuronCores).

    torchrun --nproc_per_node=8 serve.py

Same fast path as inference_neuron_i2v.py's default config (FastT5 + FastWanDiT
with both experts resident + tiled static VAE), but every model stays loaded and
the run loop is driven by requests. All ranks run in lockstep: rank 0 serves HTTP
from a background thread and hands each request to the main loop, which
broadcasts it to the other ranks, so all device work stays on each rank's main
thread. One request generates at a time; others wait in line.

The output shape (OUT_H x OUT_W, FRAME_NUM) is fixed at startup -- every graph is
compiled for it. Any input image is resized and center-cropped to it. The step
count and seed can change per request without recompiling.

    POST /generate  {"prompt": str, "image": base64 str (optional), "negative_prompt": str,
                     "seed": int, "steps": int}
        -> {"video": base64 mp4, "frames", "fps", "height", "width", "seconds", "timings"}
    GET  /health       200 while every rank has proven itself recently (see below), else 503
    GET  /readiness    200 once warm and healthy
    POST /debug/fail   {"rank": int, "mode": "hang" | "wrong"}  -- only with FAULT_INJECTION=1

Device health: a request runs collectives on every rank, so a finished request
proves all chips are alive. When the server has been idle for HEARTBEAT_S with no
such proof, the main loop runs a heartbeat instead: one tiny op + all-reduce on
every rank, checked to sum to the world size. It runs in the same loop as
requests, never alongside one, and only when the queue is empty. /health fails
if the last proof is older than HEARTBEAT_S + HEALTH_GRACE_S while idle, if a
request has run longer than MAX_REQUEST_S, or if a heartbeat returned a wrong sum.
A dead chip shows up as a collective that never returns, so the timestamp goes stale.
"""
import base64
import gc
import io
import logging
import os
import queue
import sys
import threading
import time

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [rank %(process)d]: %(message)s',
    stream=sys.stdout,
    force=True,
)
for name in ['torch', 'transformers', 'torch_neuronx', 'torch_mlir']:
    logging.getLogger(name).setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

MODEL_PATH = os.environ.get("MODEL_PATH", "/tmp/Wan2.2-I2V-A14B")
PORT = int(os.environ.get("PORT", "8000"))
FRAME_NUM = int(os.environ.get("FRAME_NUM", "21"))
OUT_H = int(os.environ.get("OUT_H", "368"))
OUT_W = int(os.environ.get("OUT_W", "480"))
NUM_STEPS = int(os.environ.get("NUM_STEPS", "10"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "50"))
FPS = int(os.environ.get("FPS", "16"))
VAE_TILE_GRID = tuple(int(v) for v in os.environ.get("VAE_TILE_GRID", "2x4").split("x"))
VAE_TILE_OVERLAP = int(os.environ.get("VAE_TILE_OVERLAP", "4"))
HEARTBEAT_S = int(os.environ.get("HEARTBEAT_S", "60"))
HEALTH_GRACE_S = int(os.environ.get("HEALTH_GRACE_S", "60"))
MAX_REQUEST_S = int(os.environ.get("MAX_REQUEST_S", "900"))
FAULT_INJECTION = os.environ.get("FAULT_INJECTION", "0") == "1"
DEFAULT_PROMPT = (
    "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. "
    "The fluffy-furred feline gazes directly at the camera with a relaxed expression. "
    "Blurred beach scenery forms the background featuring crystal-clear waters, "
    "distant green hills, and a blue sky dotted with white clouds."
)

NEURON_DEVICE = torch.device("neuron")


class Job:
    def __init__(self, req):
        self.req = req
        self.done = threading.Event()
        self.result = None
        self.error = None


def default_image():
    for name in ("examples/i2v_input.JPG", "examples/i2v_input.jpg"):
        p = os.path.join(MODEL_PATH, name)
        if os.path.exists(p):
            return Image.open(p).convert("RGB")
    return Image.new("RGB", (1280, 720), (128, 128, 128))


def fit(img, oh, ow):
    """Resize to cover (oh, ow), then center-crop."""
    scale = max(ow / img.width, oh / img.height)
    img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    x1, y1 = (img.width - ow) // 2, (img.height - oh) // 2
    return img.crop((x1, y1, x1 + ow, y1 + oh))


def to_mp4(video):
    """[3, F, H, W] in [-1, 1] -> mp4 bytes."""
    import imageio
    v = ((video.cpu().float().clamp(-1, 1) * 0.5 + 0.5) * 255).byte().permute(1, 2, 3, 0)
    buf = io.BytesIO()
    imageio.mimwrite(buf, [f.numpy() for f in v], format="mp4", fps=FPS, codec="libx264")
    return buf.getvalue()


def run_server(jobs, state):
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn

    app = FastAPI(title="Wan2.2-I2V-A14B on Trainium")

    class GenerateRequest(BaseModel):
        prompt: str
        image: str = ""
        negative_prompt: str = ""
        seed: int = 42
        steps: int = NUM_STEPS

    @app.post("/generate")
    def generate(req: GenerateRequest):
        if not state["ready"]:
            raise HTTPException(status_code=503, detail="warming up")
        if not 1 <= req.steps <= MAX_STEPS:
            raise HTTPException(status_code=400, detail=f"steps must be 1..{MAX_STEPS}")
        png = None
        if req.image:
            try:
                img = Image.open(io.BytesIO(base64.b64decode(req.image))).convert("RGB")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"bad image: {e}")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            png = buf.getvalue()
        job = Job({"type": "generate", "prompt": req.prompt, "negative_prompt": req.negative_prompt,
                   "seed": req.seed, "steps": req.steps, "png": png})
        jobs.put(job)
        job.done.wait()
        if job.error:
            raise HTTPException(status_code=500, detail=job.error)
        return job.result

    def device_health():
        """(healthy, detail) from the main loop's proof-of-life bookkeeping."""
        now = time.time()
        age = round(now - state["last_ok"], 1)
        if state["device_error"]:
            return False, {"reason": state["device_error"], "last_ok_age_s": age}
        busy = state["busy_since"]
        if busy is not None:
            running = round(now - busy, 1)
            if running > MAX_REQUEST_S:
                return False, {"reason": f"request running {running} s > {MAX_REQUEST_S} s",
                               "last_ok_age_s": age}
            return True, {"busy_s": running, "last_ok_age_s": age}
        if age > HEARTBEAT_S + HEALTH_GRACE_S:
            return False, {"reason": f"no proof of life for {age} s", "last_ok_age_s": age}
        return True, {"last_ok_age_s": age}

    @app.get("/health")
    def health():
        if not state["ready"]:
            return {"status": "starting"}      # startupProbe gates liveness until ready
        ok, detail = device_health()
        if not ok:
            logger.info(f"/health 503: {detail}")
            raise HTTPException(status_code=503, detail=detail)
        return {"status": "healthy", **detail}

    @app.get("/readiness")
    def readiness():
        if not state["ready"]:
            raise HTTPException(status_code=503, detail="warming up")
        ok, detail = device_health()
        if not ok:
            raise HTTPException(status_code=503, detail=detail)
        return {"status": "ready", "height": OUT_H, "width": OUT_W, "frames": FRAME_NUM,
                "default_steps": NUM_STEPS, **detail}

    class FailRequest(BaseModel):
        rank: int
        mode: str = "hang"

    @app.post("/debug/fail")
    def debug_fail(req: FailRequest):
        if not FAULT_INJECTION:
            raise HTTPException(status_code=404, detail="FAULT_INJECTION is off")
        if req.mode not in ("hang", "wrong") or not 0 <= req.rank < dist.get_world_size():
            raise HTTPException(status_code=400, detail="mode must be hang|wrong, rank in world")
        jobs.put(Job({"type": "fail", "rank": req.rank, "mode": req.mode}))
        logger.info(f"fault injection queued: rank {req.rank} {req.mode}")
        return {"queued": True, "rank": req.rank, "mode": req.mode}

    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")


def main():
    dist.init_process_group(backend="neuron")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.neuron.set_device(local_rank)
    rank, world_size = dist.get_rank(), dist.get_world_size()
    torch.set_grad_enabled(False)

    from wan.configs import WAN_CONFIGS
    from wan.modules.model import WanModel
    from wan.modules.t5 import T5EncoderModel
    from wan.modules.vae2_1 import Wan2_1_VAE
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
    from models.tp_utils import init_tp_group, shard_model_tp, get_tp_rank
    from models.parallel_state import init_sp_group
    from models.fast_t5 import FastT5
    from models.fast_dit import FastWanDiT
    from models.vae_static21 import StaticWanVAE
    import torchvision.transforms.functional as TF

    config = WAN_CONFIGS['i2v-A14B']
    vae_stride, patch_size, z_dim = config.vae_stride, config.patch_size, 16
    mult = vae_stride[1] * patch_size[1]
    assert OUT_H % mult == 0 and OUT_W % mult == 0, f"OUT_H/OUT_W must be multiples of {mult}"
    lat_h, lat_w = OUT_H // vae_stride[1], OUT_W // vae_stride[2]
    F = FRAME_NUM
    T_latent = (F - 1) // vae_stride[0] + 1
    boundary = config.boundary * config.num_train_timesteps

    msk = torch.ones(1, F, lat_h, lat_w)
    msk[:, 1:] = 0
    msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w).transpose(1, 2)[0]
    msk = msk.to(device=NEURON_DEVICE, dtype=torch.bfloat16)

    # ── T5 (TP over all ranks), kept resident ──
    te = T5EncoderModel(
        text_len=config.text_len, dtype=config.t5_dtype, device=torch.device('cpu'),
        checkpoint_path=os.path.join(MODEL_PATH, config.t5_checkpoint),
        tokenizer_path=os.path.join(MODEL_PATH, config.t5_tokenizer))
    t5 = FastT5(te, NEURON_DEVICE)
    del te
    gc.collect()

    # ── VAE on every rank: whole-frame encode, patch-parallel tiled decode ──
    vae = Wan2_1_VAE(vae_pth=os.path.join(MODEL_PATH, config.vae_checkpoint), device=torch.device('cpu'))
    vae.model = vae.model.to(device=NEURON_DEVICE, dtype=torch.bfloat16)
    vae.mean = vae.mean.to(device=NEURON_DEVICE, dtype=torch.bfloat16)
    vae.std = vae.std.to(device=NEURON_DEVICE, dtype=torch.bfloat16)
    vae.scale = [vae.mean, 1.0 / vae.std]
    vae = StaticWanVAE(vae)

    # ── Both DiT experts resident, TP-sharded ──
    init_tp_group(tp_degree=world_size)
    init_sp_group(tp_degree=world_size)
    tp_rank = get_tp_rank()
    grid = (T_latent, lat_h // 2, lat_w // 2)
    experts = {}
    for name, sub, g in (("high", config.high_noise_checkpoint, config.sample_guide_scale[1]),
                         ("low", config.low_noise_checkpoint, config.sample_guide_scale[0])):
        if rank == 0:
            logger.info(f"loading {name}_noise_model (resident)...")
        m = WanModel.from_pretrained(MODEL_PATH, subfolder=sub)
        m.eval().requires_grad_(False)
        shard_model_tp(m, tp_rank, world_size)
        m = m.to(torch.bfloat16).to(NEURON_DEVICE)
        experts[name] = FastWanDiT(m, grid=grid, guide_scale=g, device=NEURON_DEVICE)
    dist.barrier()
    if rank == 0:
        logger.info(f"ready to compile: {OUT_H}x{OUT_W} {F}f latent=[{z_dim}, {T_latent}, {lat_h}, {lat_w}] "
                    f"world={world_size}")

    fault = {"mode": None}

    def maybe_hang():
        if fault["mode"] == "hang":
            logger.info(f"FAULT INJECTION: rank {rank} hangs (simulated dead chip)")
            while True:
                time.sleep(3600)

    def heartbeat():
        """Tiny op + all-reduce on every rank; True if the sum equals the world size."""
        maybe_hang()
        x = torch.ones(1, dtype=torch.float32, device=NEURON_DEVICE)
        x = x * (0.0 if fault["mode"] == "wrong" else 1.0)
        dist.all_reduce(x)
        return round(x.item()) == world_size

    def generate(prompt, negative_prompt, seed, steps, img):
        """One video on every rank; rank 0 returns (mp4 bytes, timings)."""
        maybe_hang()
        t0 = time.time()
        ctx = t5.encode([prompt, negative_prompt or config.sample_neg_prompt])
        for fast in experts.values():
            fast.set_context(ctx[0:1], ctx[1:2])
        dist.barrier()
        t_text = time.time() - t0

        t1 = time.time()
        img_tensor = TF.to_tensor(fit(img, OUT_H, OUT_W)).sub_(0.5).div_(0.5)
        img_for_vae = torch.nn.functional.interpolate(
            img_tensor[None], size=(OUT_H, OUT_W), mode='bicubic').transpose(0, 1)
        img_seq = torch.cat([img_for_vae, torch.zeros(3, F - 1, OUT_H, OUT_W)], dim=1)
        z = vae.encode([img_seq.to(device=NEURON_DEVICE, dtype=torch.bfloat16)])
        y = torch.cat([msk, z[0].to(torch.bfloat16)], dim=0)
        y[0, 0, 0, 0].item()
        t_encode = time.time() - t1

        t2 = time.time()
        seed_g = torch.Generator(device=torch.device("cpu"))
        seed_g.manual_seed(seed)
        latent = torch.randn(z_dim, T_latent, lat_h, lat_w, dtype=torch.float32,
                             generator=seed_g, device=torch.device("cpu"))
        latent = latent.to(torch.bfloat16).to(NEURON_DEVICE)
        sched = FlowUniPCMultistepScheduler(
            num_train_timesteps=config.num_train_timesteps, shift=1, use_dynamic_shifting=False)
        sched.set_timesteps(steps, device=torch.device("cpu"), shift=config.sample_shift)
        for t in sched.timesteps:
            noise_pred = experts['high' if t.item() >= boundary else 'low'](latent, y, t.item())
            latent = sched.step(noise_pred.cpu().unsqueeze(0), t, latent.cpu().unsqueeze(0),
                                return_dict=False, generator=seed_g)[0].squeeze(0).to(NEURON_DEVICE)
        t_denoise = time.time() - t2

        t3 = time.time()
        videos = vae.decode_tiled([latent.to(torch.bfloat16)], grid=VAE_TILE_GRID, overlap=VAE_TILE_OVERLAP)
        mp4 = to_mp4(videos[0]) if rank == 0 else None
        dist.barrier()
        t_vae = time.time() - t3
        timings = {"text_encode": round(t_text, 2), "vae_encode": round(t_encode, 2),
                   "denoise": round(t_denoise, 2), "vae_decode": round(t_vae, 2),
                   "total": round(time.time() - t0, 2)}
        return mp4, timings

    # Warmup: compiles (or loads from cache) every graph a request uses.
    for i in range(2):
        _, timings = generate(DEFAULT_PROMPT, "", 42, NUM_STEPS, default_image())
        if rank == 0:
            logger.info(f"warmup {i + 1}/2: {timings}")

    assert heartbeat(), "warmup heartbeat: wrong all-reduce sum"
    jobs = queue.Queue()
    state = {"ready": False, "last_ok": time.time(), "busy_since": None, "device_error": None}
    if rank == 0:
        threading.Thread(target=run_server, args=(jobs, state), daemon=True).start()
    state["ready"] = True
    if rank == 0:
        logger.info(f"SERVING on :{PORT}  ({OUT_H}x{OUT_W}, {F} frames, default {NUM_STEPS} steps)")

    while True:
        job = None
        if rank == 0:
            # Wait for work until HEARTBEAT_S has passed without proof of life.
            wait = max(0.0, HEARTBEAT_S - (time.time() - state["last_ok"]))
            try:
                job = jobs.get(timeout=wait) if wait > 0 else jobs.get_nowait()
            except queue.Empty:
                pass
        msg = [job.req if job is not None else {"type": "heartbeat"}]
        dist.broadcast_object_list(msg, src=0)
        req = msg[0]

        if req["type"] == "heartbeat":
            ok = heartbeat()
            if rank == 0:
                if ok:
                    state["last_ok"] = time.time()
                else:
                    state["device_error"] = f"heartbeat all-reduce sum != {world_size}"
                    logger.info(f"HEARTBEAT FAILED: {state['device_error']}")
            continue

        if req["type"] == "fail":
            if rank == req["rank"]:
                fault["mode"] = req["mode"]
            if rank == 0:
                job.result = {"queued": True}
                job.done.set()
            continue

        if rank == 0:
            state["busy_since"] = time.time()
        try:
            img = Image.open(io.BytesIO(req["png"])).convert("RGB") if req["png"] else default_image()
            mp4, timings = generate(req["prompt"], req["negative_prompt"], req["seed"], req["steps"], img)
            if rank == 0:
                state["last_ok"] = time.time()
                logger.info(f"request seed={req['seed']} steps={req['steps']} {timings} "
                            f"prompt={req['prompt'][:80]!r}")
                job.result = {"video": base64.b64encode(mp4).decode("ascii"), "frames": F, "fps": FPS,
                              "height": OUT_H, "width": OUT_W, "seconds": timings["total"],
                              "timings": timings}
        except Exception as e:
            logger.exception("generation failed")
            if rank == 0:
                job.error = str(e)
        finally:
            if rank == 0:
                state["busy_since"] = None
                job.done.set()

if __name__ == "__main__":
    main()
