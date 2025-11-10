import streamlit as st
from diffusers import StableDiffusionInstructPix2PixPipeline
from PIL import Image
import torch

@st.cache_resource
def load_model():
    return StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix",
        safety_checker=None
    ).to("cpu")

pipe = load_model()

st.title("Sketch to Real Image Generator 🖼️")

sketch_file = st.file_uploader("Upload a Sketch", type=["jpg", "png"])
prompt = st.text_input("Enter a prompt", "color this into a realistic airplane")

if sketch_file:
    sketch_img = Image.open(sketch_file).convert("RGB").resize((512, 512))
    st.image(sketch_img, caption="Sketch Input")

    if st.button("Generate Image"):
        with st.spinner("Generating..."):
            result = pipe(
                prompt=prompt,
                image=sketch_img,
                num_inference_steps=50,
                guidance_scale=7.5
            ).images[0]
        st.image(result, caption="Generated Realistic Image")
