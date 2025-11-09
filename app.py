import streamlit as st
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from PIL import Image
import numpy as np
import torch
import cv2

st.set_page_config(page_title="Tiny Sketch-to-Image Generator", layout="wide")

# ----------------------------
# Load Tiny ControlNet Model
# ----------------------------
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
    pipe.to("cpu")  # Important for Streamlit Cloud
    return pipe

pipe = load_model()

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("✏️ Tiny Sketch-to-Image Generator (Streamlit Cloud Version)")
st.write("Upload a sketch image. The system extracts edges and generates a realistic image using a lightweight ControlNet model.")

uploaded = st.file_uploader("Upload a sketch", type=["png", "jpg", "jpeg"])
prompt = st.text_input("Prompt:", "a realistic airplane")

if uploaded:
    sketch = Image.open(uploaded).convert("RGB")

    # Convert sketch to edges using Canny
    sketch_np = np.array(sketch)
    edges = cv2.Canny(sketch_np, 100, 200)
    edges_img = Image.fromarray(edges)

    col1, col2 = st.columns(2)
    with col1:
        st.image(sketch, caption="Original Sketch", use_column_width=True)
    with col2:
        st.image(edges_img, caption="Detected Edges", use_column_width=True)

    if st.button("Generate Image"):
        with st.spinner("Generating... (this may take 20–40 seconds on CPU)"):
            result = pipe(
                prompt,
                image=edges_img,
                num_inference_steps=15,
                guidance_scale=7.5
            ).images[0]

        st.image(result, caption="Generated Output", use_column_width=True)
