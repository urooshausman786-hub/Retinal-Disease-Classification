import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Page title
st.title("👁️ Retinal Disease Classification using MobileNetV2")
st.write("Upload a retinal image to predict the disease class.")

# Load model
@st.cache_resource
def load_mobilenet_model():
    model = load_model("MobileNetV2_model.h5")
    return model

model = load_mobilenet_model()

# Upload image
uploaded_file = st.file_uploader("Choose a retinal image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).resize((224, 224))
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    # Predict
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions) * 100

    st.success(f"✅ Predicted Class: {predicted_class} | Confidence: {confidence:.2f}%")
