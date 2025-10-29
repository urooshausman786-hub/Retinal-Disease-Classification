import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
import requests
import os

# -----------------------------
# Page Setup
# -----------------------------
st.set_page_config(page_title="Retinal Disease Classifier 👁️", layout="centered")

st.title("👁️ Retinal Disease Classification")
st.markdown(
    """
    ### Upload a Retinal Image 🩺  
    This AI model analyzes the retina and predicts possible diseases.  
    Trained using **Deep Learning (MobileNetV2)** for high accuracy.
    """
)

# -----------------------------
# Download model if not present
# -----------------------------
MODEL_PATH = "retinal_model.h5"
MODEL_URL = "https://drive.google.com/uc?id=1n8zllenScXuFysusgF4OpWqMI_-kzv7y"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading AI Model... ⏳"):
        r = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
    st.success("Model downloaded successfully!")

# -----------------------------
# Load model safely
# -----------------------------
@st.cache_resource
def load_retinal_model():
    model = load_model(MODEL_PATH)
    return model

model = load_retinal_model()

# -----------------------------
# Define class names
# -----------------------------
CLASS_NAMES = [
    "Normal",
    "Diabetic Retinopathy",
    "Glaucoma",
    "Cataract",
    "Age-related Macular Degeneration"
]

# -----------------------------
# File uploader
# -----------------------------
uploaded_file = st.file_uploader("📤 Upload Retinal Image (JPG / PNG):", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="🩻 Uploaded Retinal Image", use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Predict
    with st.spinner("Analyzing image... 🔍"):
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction)

    st.success(f"### 🧠 Predicted Disease: **{CLASS_NAMES[predicted_class]}**")
    st.progress(float(confidence))
    st.caption(f"Model Confidence: **{confidence * 100:.2f}%**")

    # Display probabilities
    st.subheader("🔢 Class Probabilities:")
    for i, class_name in enumerate(CLASS_NAMES):
        st.write(f"- {class_name}: {prediction[0][i] * 100:.2f}%")
else:
    st.info("👆 Please upload a retinal image to get predictions.")
