import streamlit as st
from diffusers import StableDiffusionPipeline
import torch

# Function to load the model (cached for better performance)
@st.cache_resource
def load_model():
    model_id = "CompVis/stable-diffusion-v1-4"  # Model from Hugging Face
    with st.spinner("Loading the model... This may take a while on the first run."):
        pipe = StableDiffusionPipeline.from_pretrained(model_id)
        # Use GPU if available, otherwise fall back to CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = pipe.to(device)
    return pipe

# Streamlit App
st.title("Text-to-Image Generator")
st.markdown(
    """
    **Generate stunning images from your imagination using Stable Diffusion!**  
    Powered by [Stable Diffusion](https://github.com/CompVis/stable-diffusion) and optimized for deployment on Streamlit.
    """
)

# User input for text prompt
prompt = st.text_input("Enter a description for the image you want to generate:")

# Generate button
if st.button("Generate"):
    if prompt.strip():
        st.subheader("Generated Image")
        try:
            pipe = load_model()
            with st.spinner("Generating image..."):
                image = pipe(prompt).images[0]
            st.image(image, caption="Your Generated Image", use_column_width=True)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a description to generate an image.")

# Footer
st.markdown(
    """
    ---
    **Notes:**  
    - This app uses the Stable Diffusion model to generate images.  
    - The first run may take longer as the model downloads weights.  
    """
)
