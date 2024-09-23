#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024.4.16
# @Author  : HinGwenWong

import requests
import streamlit as st
from PIL import Image


# FastAPI Endpoint URL
DIGITAL_HUMAN_API_URL = "http://localhost:8000/generate_digital_human"

# Set up the Streamlit page configuration
st.set_page_config(
    page_title="Digital Human Generator",
    page_icon="🤖",
    layout="wide",
)

# Function to call the FastAPI digital human generation endpoint
def generate_digital_human(text_input, image_file):
    files = {'file': image_file}
    data = {'text_input': text_input}
    
    # Make the API call to FastAPI
    response = requests.post(DIGITAL_HUMAN_API_URL, data=data, files=files)
    
    if response.status_code == 200:
        st.success("Digital human generation successful!")
        return response.content  # Assuming this returns the generated video file
    else:
        st.error(f"Failed to generate digital human: {response.text}")
        return None

# Streamlit UI
st.title("Digital Human Generation")

# Text input for the script the digital human will speak
text_input = st.text_area("Enter the script for the digital human to say", "")

# File upload for the avatar or image representing the digital human
image_file = st.file_uploader("Upload the avatar image", type=["jpg", "jpeg", "png"])

if st.button("Generate Digital Human"):
    if text_input and image_file:
        st.info("Generating digital human...")
        result = generate_digital_human(text_input, image_file)
        
        if result:
            # Display the generated video
            st.video(result)
    else:
        st.error("Please provide both a script and an avatar image.")
