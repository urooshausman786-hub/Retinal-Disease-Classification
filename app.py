import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# -------------------------------
# APP CONFIGURATION
# -------------------------------
st.set_page_config(page_title="Retinal Disease Classifier", page_icon="🧠", layout="wide")

st.title("🩺 Retinal Disease Classification using Deep Learning")
st.write("Upload a retinal fundus image and let the model predict the disease type.")

# -------------------------------
# DOWNLOAD MODEL FROM GOOGLE DRIVE
# -------------------------------
MODEL_PATH = "retinal_model.h5"

# 👇 Replace the link below with YOUR Google Drive file ID
# Example link: https://drive.google.com/file/d/1n8zllenScXuFysusgF4OpWqMI_-kzv7y/view?usp=sharing
file_id = "1n8zllenScXuFysusgF4OpWqMI_-kzv7y"
download_url = f"https://drive.google.com/uc?id={file_id}"

if not os.path.exists(MODEL_PATH):
    with st.spinner("📥 Downloading model from Google Drive... (this may take a minute)"):
        gdown.download(download_url, MODEL_PATH, quiet=False)

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# -------------------------------
# CLASS NAMES (edit as per your dataset)
# -------------------------------
CLASS_NAMES = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']

# -------------------------------
# UPLOAD SECTION
# -------------------------------
uploaded_file = st.file_uploader("📤 Upload a retinal image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Make prediction
    with st.spinner("🔍 Analyzing image..."):
        prediction = model.predict(img_array)
        confidence = np.max(prediction)
        predicted_class = CLASS_NAMES[np.argmax(prediction)]

    # Display results
    st.success(f"✅ **Predicted Disease:** {predicted_class}")
    st.info(f"📊 Confidence: {confidence * 100:.2f}%")

    st.markdown("---")
    st.write("**Note:** This app is for academic demonstration only.")
else:
    st.warning("Please upload a retinal image to get prediction.")
