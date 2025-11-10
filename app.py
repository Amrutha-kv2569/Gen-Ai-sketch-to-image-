import streamlit as st
from openai import OpenAI
from PIL import Image
import io

st.title("Sketch → Real Image Generator 🖼️ (GPT-powered)")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

sketch_file = st.file_uploader("Upload a Sketch", type=["jpg", "png"])
prompt = st.text_input("Describe how the sketch should be transformed",
                       "color this into a realistic airplane")

if sketch_file:
    sketch_img = Image.open(sketch_file).convert("RGB")
    st.image(sketch_img, caption="Sketch Input")

    if st.button("Generate Image"):
        with st.spinner("Generating..."):

            # Convert PIL image to bytes
            buf = io.BytesIO()
            sketch_img.save(buf, format="PNG")
            img_bytes = buf.getvalue()

            result = client.images.edit(
                model="gpt-image-1",
                prompt=prompt,
                image=img_bytes
            )

            output_url = result.data[0].url

        st.image(output_url, caption="Generated Image")
