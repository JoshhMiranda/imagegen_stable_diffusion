import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from io import BytesIO
from PIL import Image
import base64
from authtoken import auth_token

st.title("Image generation app")
st.write("Let your imagination run free using Stable Diffusion!")


prompt = st.text_input("Enter your thoughts....", value= "a car driving by the scenic French Rivieria", max_chars=200)


# Load the Stable Diffusion model
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"


# Load the model to the available device (GPU/CPU)
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, revision="fp16", torch_dtype=torch.float16 if device == "cuda" else torch.float32, use_auth_token=auth_token
)
pipe.to(device)



# Generate the image when the button is clicked
if st.button("Generate Image"):
    if prompt:
        with torch.autocast(device):
            # Generate the image from the text prompt
            image = pipe(prompt, guidance_scale=8.5).images[0]

        # Save the generated image temporarily in memory
        # buffer = BytesIO()
        # image.save(buffer, format="PNG")
        # buffer.seek(0)

        # Display the image in Streamlit
        st.image(image, caption="Generated Image", use_column_width=True)
    else:
        st.error("Please enter a valid prompt to generate an image :D")