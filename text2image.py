import streamlit as st
from diffusers import StableDiffusionPipeline
import torch

# Load Stable Diffusion model
@st.cache_resource
def load_model():
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe

st.title("Text-to-Image Generator")
st.write("Generate images from your imagination using Stable Diffusion!")

# Input box for the user prompt
prompt = st.text_input("Enter a description for the image you want to generate:")

# Generate image button
if st.button("Generate"):
    if prompt:
        with st.spinner("Generating image..."):
            pipe = load_model()
            image = pipe(prompt).images[0]
            st.image(image, caption="Generated Image", use_column_width=True)
    else:
        st.warning("Please enter a description.")

# Footer
st.write("Powered by [Stable Diffusion](https://github.com/CompVis/stable-diffusion)")
