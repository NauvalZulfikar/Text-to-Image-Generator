import streamlit as st
from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image

# Load the Stable Diffusion pipeline
@st.cache_resource
def load_pipeline():
    return StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    ).to("cuda" if torch.cuda.is_available() else "cpu")

def main():
    st.title("Camera Capture with Stable Diffusion")
    pipeline = load_pipeline()

    # Step 1: Capture an image using the device camera
    captured_image = st.camera_input("Take a picture")
    
    # Step 2: Text input for Stable Diffusion
    if captured_image is not None:
        st.image(captured_image, caption="Captured Image", use_column_width=True)
        
        # Text prompt for Stable Diffusion
        prompt = st.text_input("Enter a description for the image transformation (e.g., 'a futuristic cityscape')", "a dreamlike scene in nature")
        
        # Button to generate AI-modified image
        if st.button("Generate AI Image"):
            with st.spinner("Generating image..."):
                # Open the captured image
                original_image = Image.open(captured_image)
                
                # Resize the image to the model's expected input size
                original_image = original_image.resize((512, 512))

                # Generate the transformed image
                generated_image = pipeline(prompt=prompt, image=original_image, strength=0.75, guidance_scale=7.5).images[0]
                
                # Display the generated image
                st.image(generated_image, caption="AI-Generated Image", use_column_width=True)
                
                # Optional: Save the generated image
                generated_image.save("ai_generated_image.jpg")
                st.success("AI-generated image saved as 'ai_generated_image.jpg'")

if __name__ == "__main__":
    main()
