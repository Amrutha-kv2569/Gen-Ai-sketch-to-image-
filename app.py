import streamlit as st
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from PIL import Image
import numpy as np
import cv2
import torch

st.set_page_config(page_title="Tiny Sketch-to-Image Demo")

@st.cache_resource
def load_model():
    controlnet = ControlNetModel.from_pretrained(
        "damo/cv_tinynas_controlnet_canny",
        torch_dtype=torch.float32
    )

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float32
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cpu")
    return pipe

pipe = load_model()

st.title("✏️ Tiny Sketch-to-Image Generator (Streamlit Cloud Version)")

uploaded = st.file_uploader("Upload sketch", type=["png","jpg","jpeg"])
prompt = st.text_input("Prompt", "a realistic airplane")

if uploaded:
    sketch = Image.open(uploaded).convert("RGB")

    # Convert to edges
    sketch_np = np.array(sketch)
    edges = cv2.Canny(sketch_np, 100, 200)
    edges_img = Image.fromarray(edges)

    st.image(edges_img, caption="Detected Edges (Input to Tiny ControlNet)")

    if st.button("Generate"):
        with st.spinner("Generating image... (CPU, 20–40s)"):
            result = pipe(
                prompt,
                image=edges_img,
                num_inference_steps=15,
                guidance_scale=7.5,
            ).images[0]
        
        st.image(result, caption="Generated Output")
