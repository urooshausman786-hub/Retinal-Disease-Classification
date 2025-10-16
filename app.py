import streamlit as st
from PIL import Image

# App title
st.title("Retinal Disease Classification (Demo)")
st.write("Upload a retinal image to see a demo prediction.")

# Image uploader
uploaded_file = st.file_uploader("Choose a retinal image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    # Open and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Dummy prediction
    st.write("Predicted Disease: **Normal (Demo)**")
    st.info("This is a placeholder prediction. Real model predictions will appear here once integrated.")

