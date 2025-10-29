import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import requests
from PIL import Image

# --- Page Configuration ---
st.set_page_config(page_title="Retinal Disease Classifier", layout="wide")
st.title("👁️ Retinal Disease Classification using Deep Learning")
st.write("Upload a retinal image to classify the disease using a trained CNN model.")

# --- Download Model from Google Drive if not present ---
model_path = "retinal_model.h5"

if not os.path.exists(model_path):
    with st.spinner("Downloading model, please wait..."):
        url = "https://drive.google.com/file/d/1n8zllenScXuFysusgF4OpWqMI_-kzv7y/view?usp=sharing"
        response = requests.get(url)
        with open(model_path, "wb") as f:
            f.write(response.content)
    st.success("✅ Model downloaded successfully!")

# --- Load Model ---
@st.cache_resource
def load_trained_model():
    model = load_model(model_path, compile=False)
    return model

model = load_trained_model()
st.success("✅ Model loaded successfully!")

# --- Define class names (update if you have different ones) ---
CLASS_NAMES = [
    "Diabetic Retinopathy",
    "Glaucoma",
    "Cataract",
    "Normal Retina",
    "Age-related Macular Degeneration"
]

# --- Image Upload Section ---
uploaded_file = st.file_uploader("📤 Upload a retinal image (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    image_pil = Image.open(uploaded_file).convert("RGB")
    st.image(image_pil, caption="Uploaded Retinal Image", use_column_width=True)

    # --- Preprocess the image ---
    img = image_pil.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    # --- Make prediction ---
    with st.spinner("Analyzing image..."):
        predictions = model.predict(img_array)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0]) * 100

    # --- Display results ---
    st.markdown("---")
    st.markdown(f"### 🩺 **Predicted Disease:** `{predicted_class}`")
    st.markdown(f"**Confidence:** {confidence:.2f}%")

    # --- Optional: Show probabilities for all classes ---
    st.markdown("#### 🔍 Prediction Probabilities:")
    for i, disease in enumerate(CLASS_NAMES):
        st.write(f"{disease}: {predictions[0][i]*100:.2f}%")

# --- Footer ---
st.markdown("---")
st.markdown("👩‍💻 **Developed by Uroosha Usman (MSc Computer Science, Lucknow University)**")



