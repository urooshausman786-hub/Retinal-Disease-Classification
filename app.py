
import streamlit as st
import numpy as np
from PIL import Image
import os
import gdown
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

# -----------------------------------------------
# 🔽 Step 1: Download model from Google Drive (auto)
# -----------------------------------------------
MODEL_URL = "https://drive.google.com/uc?id=1n8zllenScXuFysusgF4OpWqMI_-kzv7y"
MODEL_PATH = "retinal_model.h5"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading retinal model... please wait ⏳"):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    st.success("✅ Model downloaded successfully!")

# -----------------------------------------------
# 🧠 Step 2: Load the trained model
# -----------------------------------------------
@st.cache_resource
def load_retinal_model():
    model = load_model(MODEL_PATH)
    return model

model = load_retinal_model()

# -----------------------------------------------
# 🏷️ Step 3: Define class labels
# -----------------------------------------------
CLASS_NAMES = ["Normal", "Diabetic Retinopathy", "Glaucoma", "Cataract", "Age-related Macular Degeneration"]

# -----------------------------------------------
# 🖥️ Step 4: Streamlit App UI
# -----------------------------------------------
st.title("👁️ Retinal Disease Classification")
st.write("Upload a retinal image to get an AI-based disease prediction.")

# File uploader
uploaded_file = st.file_uploader("📤 Choose a retinal image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='🩺 Uploaded Retinal Image', use_column_width=True)

    # Preprocess for model
    img = image.resize((224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # normalization

    # Prediction
    with st.spinner("Analyzing the image..."):
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction)

    # Display result
    st.success(f"🧾 **Predicted Disease:** {CLASS_NAMES[predicted_class]}")
    st.write(f"🎯 **Confidence:** {confidence * 100:.2f}%")

    # Detailed class probabilities
    st.subheader("📊 Class Probabilities")
    for i, class_name in enumerate(CLASS_NAMES):
        st.write(f"{class_name}: {prediction[0][i] * 100:.2f}%")
else:
    st.info("Please upload an image to start prediction.")

# -----------------------------------------------
# ✅ Step 5: Instructions for deployment
# -----------------------------------------------
st.markdown("""
---
### 🚀 Deployment Instructions
1. Do **not upload** the `.h5` model file to GitHub — this app auto-downloads it from Google Drive.
2. Make sure your `requirements.txt` includes:


