import os
import time
import random
import io
import numpy as np
import streamlit as st
from PIL import Image
import torch
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)

# -----------------------------
# UI: Title & Intro
# -----------------------------
st.set_page_config(page_title="Sketch → Real Image (Stable Diffusion + ControlNet)", layout="centered")
st.title("🎨 Sketch → Real Image Generator")
st.caption("Stable Diffusion v1.5 + ControlNet (Scribble). Upload a sketch and add a prompt to generate a realistic image.")

# -----------------------------
# Utilities
# -----------------------------
def to_rgb_square(img: Image.Image, size: int = 512) -> Image.Image:
    """Convert to RGB + resize to square with minimal distortion."""
    img = img.convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.BICUBIC)
    return img

@st.cache_resource(show_spinner=True)
def load_pipeline(low_vram: bool = True, prefer_fp16: bool = True):
    """
    Load ControlNet (scribble) + SD v1.5 pipeline, cached across reruns.
    low_vram: enables cpu offload/attention slicing if True
    prefer_fp16: uses float16 when CUDA available
    """
    dtype = torch.float16 if (torch.cuda.is_available() and prefer_fp16) else torch.float32

    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_scribble",
        torch_dtype=dtype
    )

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None  # disable HF safety checker for speed; handle images responsibly
    )

    # Scheduler
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        if low_vram:
            # Reduce VRAM usage
            pipe.enable_attention_slicing()
            pipe.enable_model_cpu_offload()
    else:
        # CPU-only path
        if low_vram:
            pipe.enable_attention_slicing()

    return pipe

def seed_everything(seed: int | None):
    if seed is None or seed < 0:
        seed = random.randint(0, 2**31 - 1)
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    random.seed(seed)
    return seed

# -----------------------------
# Sidebar Controls
# -----------------------------
with st.sidebar:
    st.subheader("⚙️ Generation Settings")
    resolution = st.selectbox("Resolution", [256, 384, 512], index=2)
    steps = st.slider("Inference steps", 10, 75, 30, step=1)
    guidance = st.slider("Guidance scale (CFG)", 1.0, 15.0, 8.0, step=0.1)
    ctrl_scale = st.slider("Control strength", 0.1, 2.0, 1.0, step=0.05, help="Higher = follow sketch more strictly")
    seed = st.number_input("Seed (-1 for random)", value=-1, step=1)
    low_vram = st.toggle("Low VRAM mode", value=True, help="Enable attention slicing & CPU offload (useful on small GPUs/CPU).")
    prefer_fp16 = st.toggle("Prefer FP16 on GPU", value=True)

    st.markdown("---")
    st.caption("Tip: Lower resolution & steps if you hit memory limits.")

# -----------------------------
# Load model (cached)
# -----------------------------
with st.spinner("Loading Stable Diffusion + ControlNet…"):
    pipe = load_pipeline(low_vram=low_vram, prefer_fp16=prefer_fp16)

# -----------------------------
# Main Inputs
# -----------------------------
col1, col2 = st.columns(2)
with col1:
    uploaded = st.file_uploader("📤 Upload a sketch (PNG/JPG)", type=["png", "jpg", "jpeg"])
with col2:
    prompt = st.text_input("🗣️ Prompt", "a realistic airplane in a blue sky, high detail, photorealistic")
neg_prompt = st.text_input("🚫 Negative prompt (optional)", "low quality, blurry, deformed, extra limbs")

# Example sketch preview & generate button
if uploaded is not None:
    try:
        sketch_img = Image.open(uploaded)
    except Exception as e:
        st.error(f"Could not open image: {e}")
        st.stop()

    sketch_disp = to_rgb_square(sketch_img, resolution)
    st.image(sketch_disp, caption="Sketch (resized)", use_container_width=True)

    if st.button("🚀 Generate"):
        # Seeding
        seed_val = seed_everything(int(seed))
        st.caption(f"Using seed: {seed_val}")

        # Prepare input
        control_image = to_rgb_square(sketch_img, resolution)

        # Inference
        t0 = time.time()
        generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed_val)
        try:
            with st.spinner("Generating…"):
                out = pipe(
                    prompt=prompt,
                    image=control_image,
                    negative_prompt=neg_prompt if neg_prompt.strip() else None,
                    num_inference_steps=int(steps),
                    guidance_scale=float(guidance),
                    controlnet_conditioning_scale=float(ctrl_scale),
                    generator=generator,
                )
                result = out.images[0]
        except Exception as e:
            st.error(f"Generation failed: {e}")
            st.stop()

        dt = time.time() - t0
        st.success(f"Done in {dt:.1f}s")
        st.image(result, caption="Generated image", use_container_width=True)

        # Download button
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        st.download_button("⬇️ Download PNG", data=buf.getvalue(), file_name="generated.png", mime="image/png")
else:
    st.info("Upload a sketch image to start.")

# -----------------------------
# Footer: Notes / Help
# -----------------------------
with st.expander("Notes & Tips"):
    st.markdown(
        """
- This app uses **Stable Diffusion v1.5** with **ControlNet (Scribble)** to respect your sketch structure while adding realistic detail via the prompt.
- If you see out-of-memory errors, switch **Low VRAM mode ON**, reduce **Resolution** and **Steps**.
- Results vary with the **prompt** — try describing color, lighting, background, and style (e.g., *“sunset lighting, metallic texture, cinematic look”*).
- For batch evaluation against your dataset (sketch ↔ real pairs), run a notebook to loop over sketches and compute metrics like **SSIM/PSNR/FID**.
        """
    )
