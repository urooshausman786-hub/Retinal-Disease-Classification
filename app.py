import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
import gdown
import os

st.set_page_config(page_title="Retinal Disease Classifier", layout="centered")

# -----------------------------
# Download model from Google Drive if not exists
# -----------------------------
MODEL_PATH = "retinal_model.h5"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model file... please wait ⏳"):
        url = "https://drive.google.com/uc?id=1n8zllenScXuFysusgF4OpWqMI_-kzv7y"  # ✅ Replace with your file ID
        gdown.download(url, MODEL_PATH, quiet=False)

# -----------------------------
# Load model safely
# -----------------------------
@st.cache_resource
def load_retinal_model():
    model = load_model(MODEL_PATH)
    return model

model = load_retinal_model()

# -----------------------------
# Class Labels
# -----------------------------
CLASS_NAMES = ["Normal", "Diabetic Retinopathy", "Glaucoma", "Cataract", "Age-related Macular Degeneration"]

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("👁️ Retinal Disease Classification")
st.markdown("Upload a **retinal image** to detect possible disease using a trained deep learning model.")

uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Retinal Image", use_column_width=True)

    img = image.resize((224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    st.success(f"**Prediction:** {CLASS_NAMES[predicted_class]}")
    st.write(f"**Confidence:** {confidence * 100:.2f}%")

    st.subheader("Prediction Probabilities:")
    for i, class_name in enumerate(CLASS_NAMES):
        st.write(f"- {class_name}: {prediction[0][i] * 100:.2f}%")

else:
    st.info("Please upload a retinal image to start prediction.")
