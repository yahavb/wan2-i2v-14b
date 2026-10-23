"""Gradio front end for serve.py: upload an image, type a prompt, get the video,
then let Muse-Glimmer-30B (vLLM) describe the video and score how well it matches the prompt.

    WAN_URL=http://wan2-i2v-14b:8000 VLM_URL=http://muse-glimmer-30b-vllm:8000/v1 python gradio_app.py

Score: cosine similarity between sentence embeddings of the prompt and of the video
description, computed on this pod's CPU (sentence-transformers, EMBED_MODEL).
"""
import base64
import os
import tempfile
import time

import gradio as gr
import requests

WAN_URL = os.environ.get("WAN_URL", "http://wan2-i2v-14b.default.svc.cluster.local:8000")
VLM_URL = os.environ.get("VLM_URL", "http://muse-glimmer-30b-vllm.default.svc.cluster.local:8000/v1")
VLM_MODEL = os.environ.get("VLM_MODEL", "meta-models/Muse-Glimmer-30B")
VLM_MAX_TOKENS = int(os.environ.get("VLM_MAX_TOKENS", "384"))
EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_VLM_PROMPT = "Describe what happens in this video in one or two sentences."
DEFAULT_PROMPT = (
    "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. "
    "The fluffy-furred feline gazes directly at the camera with a relaxed expression. "
    "Blurred beach scenery forms the background featuring crystal-clear waters, "
    "distant green hills, and a blue sky dotted with white clouds."
)
# The server's fallback start image (MODEL_PATH/examples/i2v_input.JPG), shown so it is visible.
DEFAULT_IMAGE_URL = "https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B/resolve/main/examples/i2v_input.JPG"


def default_image():
    path = os.path.join(tempfile.gettempdir(), "wan_default_i2v_input.jpg")
    if not os.path.exists(path):
        try:
            r = requests.get(DEFAULT_IMAGE_URL, timeout=30)
            r.raise_for_status()
            with open(path, "wb") as f:
                f.write(r.content)
        except requests.RequestException as e:
            print(f"default image download failed: {e}")
            return None
    return path


_embedder = None


def embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
    return _embedder


def similarity(a, b):
    """Cosine similarity of the two texts' sentence embeddings, in percent."""
    ea, eb = embedder().encode([a, b], normalize_embeddings=True)
    return round(float(ea @ eb) * 100, 1)


def describe_video(mp4_b64, question):
    """Muse-Glimmer-30B on vLLM: the mp4 goes in as-is (it samples 8 frames, 480x368)."""
    body = {"model": VLM_MODEL, "temperature": 0, "max_tokens": VLM_MAX_TOKENS,
            "messages": [{"role": "system", "content": "Reasoning strength: low"},
                         {"role": "user", "content": [
                             {"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + mp4_b64}},
                             {"type": "text", "text": question}]}]}
    r = requests.post(f"{VLM_URL}/chat/completions", json=body, timeout=600)
    r.raise_for_status()
    reply = r.json()["choices"][0]["message"]["content"]
    if "to=user" not in reply:
        raise RuntimeError(f"no final answer within {VLM_MAX_TOKENS} tokens (only reasoning); "
                           "raise VLM_MAX_TOKENS")
    return reply.rsplit("to=user", 1)[-1].strip()     # drop the model's reasoning channel


def server_info():
    try:
        r = requests.get(f"{WAN_URL}/readiness", timeout=5)
        if r.status_code == 200:
            i = r.json()
            return f"server ready: {i['width']}x{i['height']}, {i['frames']} frames"
        return "server warming up (compiling) - try again in a few minutes"
    except requests.RequestException as e:
        return f"server unreachable: {e}"


def generate(image, prompt, negative_prompt, seed, steps, vlm_prompt):
    """Yields (video, status, description, score) as each stage finishes."""
    body = {"prompt": prompt, "negative_prompt": negative_prompt, "seed": int(seed), "steps": int(steps)}
    if image:
        with open(image, "rb") as f:
            body["image"] = base64.b64encode(f.read()).decode("ascii")
    yield None, "generating video...", "", None
    r = requests.post(f"{WAN_URL}/generate", json=body, timeout=1800)
    if r.status_code != 200:
        raise gr.Error(f"{r.status_code}: {r.text[:300]}")
    out = r.json()
    path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    with open(path, "wb") as f:
        f.write(base64.b64decode(out["video"]))
    t = out["timings"]
    status = (f"video: {out['frames']} frames {out['width']}x{out['height']} in {t['total']} s "
              f"(text {t['text_encode']} s, encode {t['vae_encode']} s, denoise {t['denoise']} s, "
              f"decode {t['vae_decode']} s)")
    yield path, status + "\nMuse-Glimmer is watching the video...", "", None

    try:
        t0 = time.time()
        desc = describe_video(out["video"], vlm_prompt)
        t_vlm = time.time() - t0
        t0 = time.time()
        score = similarity(prompt, desc)
        t_sim = time.time() - t0
    except Exception as e:
        yield path, status + f"\nvideo understanding failed: {e}", "", None
        return
    yield path, status + f"\nunderstanding: {t_vlm:.1f} s, similarity: {t_sim:.2f} s (CPU)", desc, score

embedder()     # download + load the CPU embedding model before the first request

with gr.Blocks(title="Wan2.2 image-to-video and Muse-Glimmer-30B on AWS Trainium") as demo:
    gr.Markdown("# Wan2.2 image-to-video and Muse-Glimmer-30B on AWS Trainium\nThe video starts from the image below. "
                "Replace it with your own, write a prompt, press **Generate**. Muse-Glimmer-30B then "
                "describes the video, and the score shows how close that description is to your prompt.")
    with gr.Row():
        with gr.Column():
            image = gr.Image(value=default_image(), label="Start image (default: model example)",
                             type="filepath", height=300)
            prompt = gr.Textbox(label="Prompt", value=DEFAULT_PROMPT, lines=4)
            negative = gr.Textbox(label="Negative prompt (empty = model default)", lines=2)
            with gr.Row():
                seed = gr.Number(label="Seed", value=42, precision=0)
                steps = gr.Slider(label="Steps", minimum=1, maximum=50, value=10, step=1)
            vlm_prompt = gr.Textbox(label="Question for Muse-Glimmer", value=DEFAULT_VLM_PROMPT, lines=2)
            btn = gr.Button("Generate", variant="primary")
        with gr.Column():
            video = gr.Video(label="Video", autoplay=True, loop=True, height=420)
            status = gr.Textbox(label="Status", interactive=False, lines=3)
            description = gr.Textbox(label="What Muse-Glimmer sees in the video", interactive=False, lines=4)
            score = gr.Number(label="Prompt vs description cosine similarity (%)", interactive=False)
    demo.load(server_info, None, status)
    btn.click(generate, [image, prompt, negative, seed, steps, vlm_prompt], [video, status, description, score])

demo.queue().launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")),
                    root_path=os.environ.get("ROOT_PATH", ""))
