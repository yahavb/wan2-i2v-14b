"""Gradio front end for serve.py: upload an image, type a prompt, get the video.

    WAN_URL=http://wan2-i2v-14b:8000 python gradio_app.py
"""
import base64
import os
import tempfile

import gradio as gr
import requests

WAN_URL = os.environ.get("WAN_URL", "http://wan2-i2v-14b.default.svc.cluster.local:8000")
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


def server_info():
    try:
        r = requests.get(f"{WAN_URL}/readiness", timeout=5)
        if r.status_code == 200:
            i = r.json()
            return f"server ready: {i['width']}x{i['height']}, {i['frames']} frames"
        return "server warming up (compiling) - try again in a few minutes"
    except requests.RequestException as e:
        return f"server unreachable: {e}"


def generate(image, prompt, negative_prompt, seed, steps):
    body = {"prompt": prompt, "negative_prompt": negative_prompt, "seed": int(seed), "steps": int(steps)}
    if image:
        with open(image, "rb") as f:
            body["image"] = base64.b64encode(f.read()).decode("ascii")
    r = requests.post(f"{WAN_URL}/generate", json=body, timeout=1800)
    if r.status_code != 200:
        raise gr.Error(f"{r.status_code}: {r.text[:300]}")
    out = r.json()
    path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    with open(path, "wb") as f:
        f.write(base64.b64decode(out["video"]))
    t = out["timings"]
    status = (f"{out['frames']} frames {out['width']}x{out['height']} in {t['total']} s "
              f"(text {t['text_encode']} s, encode {t['vae_encode']} s, denoise {t['denoise']} s, "
              f"decode {t['vae_decode']} s)")
    return path, status


with gr.Blocks(title="Wan2.2-I2V-A14B on AWS Trainium") as demo:
    gr.Markdown("# Wan2.2 image-to-video on AWS Trainium\nThe video starts from the image below. "
                "Replace it with your own, write a prompt, press **Generate**.")
    with gr.Row():
        with gr.Column():
            image = gr.Image(value=default_image(), label="Start image (default: model example)",
                             type="filepath", height=300)
            prompt = gr.Textbox(label="Prompt", value=DEFAULT_PROMPT, lines=4)
            negative = gr.Textbox(label="Negative prompt (empty = model default)", lines=2)
            with gr.Row():
                seed = gr.Number(label="Seed", value=42, precision=0)
                steps = gr.Slider(label="Steps", minimum=1, maximum=50, value=10, step=1)
            btn = gr.Button("Generate", variant="primary")
        with gr.Column():
            video = gr.Video(label="Video", autoplay=True, loop=True, height=420)
            status = gr.Textbox(label="Status", interactive=False)
    demo.load(server_info, None, status)
    btn.click(generate, [image, prompt, negative, seed, steps], [video, status])

demo.queue().launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")),
                    root_path=os.environ.get("ROOT_PATH", ""))
