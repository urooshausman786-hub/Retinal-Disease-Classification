import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

# ---------------------- Page Config ----------------------
st.set_page_config(page_title="🩺 Retinal Disease Classifier", layout="centered")

st.title("🩺 Retinal Disease Classifier")
st.write("Upload a retinal image and get predictions using your trained MobileNetV2 TFLite model.")

# ---------------------- Model Setup ----------------------
MODEL_PATH = "MobileNetV2_model.tflite"
INPUT_SIZE = (224, 224)
LABELS = ["Normal", "Diabetic Retinopathy", "Glaucoma", "Cataract", "Age-related Macular Degeneration"]  # ✅ change if needed

# ---------------------- Load Model ----------------------
@st.cache_resource
def load_tflite_model(model_path):
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

interpreter = load_tflite_model(MODEL_PATH)

if interpreter is None:
    st.stop()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ---------------------- Preprocessing ----------------------
def preprocess_image(img):
    img = img.convert("RGB").resize(INPUT_SIZE)
    x = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(x, axis=0)

# ---------------------- Prediction ----------------------
def predict(image):
    data = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], data)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0]
    preds = np.exp(preds - np.max(preds))  # softmax
    preds = preds / np.sum(preds)
    return preds

# ---------------------- Upload Section ----------------------
uploaded_file = st.file_uploader("📤 Upload a retinal image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="🩻 Uploaded Image", use_column_width=True)

    preds = predict(image)

    st.subheader("🔍 Prediction Results:")
    for i, p in enumerate(preds):
        st.write(f"{LABELS[i] if i < len(LABELS) else f'Class {i}'}: **{p*100:.2f}%**")

    st.success(f"✅ Predicted: **{LABELS[np.argmax(preds)]}**")
