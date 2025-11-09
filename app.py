import streamlit as st
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import UniPCMultistepScheduler
from PIL import Image
import torch
import numpy as np
import cv2

st.set_page_config(page_title="Tiny Sketch-to-Image Demo")

# ----------------------------
# Load TINY ControlNet
# ----------------------------
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
    pipe.to(device)

    return pipe

pipe = load_model()

st.title("✏️ Tiny Sketch-to-Image Generator (Fast CPU Demo)")
st.write("Upload a sketch. The system converts it into an edge map and generates a realistic image using a tiny ControlNet model.")

# Upload sketch
uploaded = st.file_uploader("Upload sketch", type=["png","jpg","jpeg"])

prompt = st.text_input("Prompt", "a realistic airplane")

if uploaded:
    sketch = Image.open(uploaded).convert("RGB")

    # Convert sketch to edges
    sketch_np = np.array(sketch)
    edges = cv2.Canny(sketch_np, 100, 200)
    edges = Image.fromarray(edges)

    st.image(edges, caption="Detected edges", use_column_width=True)

    if st.button("Generate Image"):
        with st.spinner("Generating..."):
            result = pipe(
                prompt,
                image=edges,
                num_inference_steps=15,
                guidance_scale=7.5
            ).images[0]

        st.image(result, caption="Generated Image")
