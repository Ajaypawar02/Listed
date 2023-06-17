import streamlit as st
from PIL import Image
import sys
import cv2
import numpy as np
sys.path.append('./src')
from ImageCaption import ImageCaption
from tempfile import NamedTemporaryFile

st.markdown(
    """
    <style>
    .stApp {
        background-color: black;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
        """
        <style>
        body {
            background-color: #f5f5f5; /* Set the background color */
            font-family: Arial, sans-serif; /* Set the font family */
            padding: 20px; /* Add padding to the content */
        }
        .stButton button { 
            background-color: #336699; /* Set the button background color */
            color: white; /* Set the button text color */
            padding: 8px 16px; /* Adjust button padding */
            border-radius: 4px; /* Add button border radius */
            border: none; /* Remove button border */
            cursor: pointer; /* Add cursor pointer on hover */
        }
        </style>
        """,
        unsafe_allow_html=True
    )
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    file_bytes = uploaded_file.read()
    nparr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    num_captions = st.selectbox("Select the number of captions to generate", [1, 2, 3, 4, 5])


    if st.button("Generate Captions"):
        # Preprocess the image
        with st.spinner("Generating captions..."):
            ImgCap = ImageCaption()
            captions = ImgCap.photo_upload(image, num_captions)
            st.success("Captions generated successfully!")
        
        for i, caption in enumerate(captions):
            st.info(f"Caption {i+1}: {caption}")