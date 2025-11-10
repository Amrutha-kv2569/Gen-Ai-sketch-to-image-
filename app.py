import streamlit as st
import os
from diffusers import StableDiffusionInstructPix2PixPipeline
from huggingface_hub import login
from PIL import Image
import torch

# -----------------------------------------------------
# Environment settings (prevents cloud download errors)
# -----------------------------------------------------
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# Optional: If you set a HuggingFace token in Streamlit Secrets
if "HUGGINGFACE_TOKEN" in st.secrets:
    login(st.secrets["HUGGINGFACE_TOKEN"])

# -----------------------------------------------------
# Load model once
# -----------------------------------------------------
@st.cache_resource
def load_model():
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix-800m",   # ✅ smaller model for Streamlit Cloud
        safety_checker=None
    )
    pipe = pipe.to("cpu")
    return pipe

pipe = load_model()

# -----------------------------------------------------
# UI
# -----------------------------------------------------
st.title("Sketch → Real Image Generator 🖼️")
st.write("Upload a sketch and describe how you want it transformed.")

sketch_file = st.file_uploader("Upload a Sketch", type=["jpg", "png"])
prompt = st.text_input("Enter a prompt", "color this into a realistic airplane")

# -----------------------------------------------------
# Processing
# -----------------------------------------------------
if sketch_file:
    sketch_img = Image.open(sketch_file).convert("RGB").resize((512, 512))
    st.image(sketch_img, caption="Sketch Input", use_column_width=True)

    if st.button("Generate Image"):
        with st.spinner("Generating... please wait (this may take 20–40 seconds on CPU)..."):

            result = pipe(
                prompt=prompt,
                image=sketch_img,
                num_inference_steps=40,   # slightly reduced for speed
                guidance_scale=7.5
            ).images[0]

        st.image(result, caption="Generated Realistic Image", use_column_width=True)
