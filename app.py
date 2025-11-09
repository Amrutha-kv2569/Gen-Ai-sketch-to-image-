# app.py
import streamlit as st
import torch
from PIL import Image, ImageOps
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

@st.cache_resource
def load_pipe():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_scribble", torch_dtype=dtype
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    if device == "cuda":
        pipe.to("cuda")
        try: pipe.enable_xformers_memory_efficient_attention()
        except: pass
    else:
        pipe.enable_attention_slicing()
        pipe.enable_sequential_cpu_offload()
    return pipe

st.title("✏️→🖼️ Sketch-to-Image (ControlNet)")
pipe = load_pipe()

sketch = st.file_uploader("Upload sketch (png/jpg)", type=["png","jpg","jpeg"])
prompt = st.text_input("Prompt", "a realistic airplane, detailed, natural lighting")
neg = st.text_input("Negative prompt", "blurry, low quality, deformed, artifacts, watermark")
steps = st.slider("Inference steps", 10, 50, 30)
guidance = st.slider("Guidance scale", 3.0, 12.0, 8.0)
ctrl_scale = st.slider("Control strength", 0.2, 1.5, 1.0)

if sketch and st.button("Generate"):
    img = Image.open(sketch).convert("RGB").resize((512,512))
    with st.spinner("Generating..."):
        out = pipe(
            prompt=prompt, negative_prompt=neg, image=img,
            num_inference_steps=int(steps), guidance_scale=float(guidance),
            controlnet_conditioning_scale=float(ctrl_scale)
        ).images[0]
    st.image(out, caption="Generated Image", use_column_width=True)
