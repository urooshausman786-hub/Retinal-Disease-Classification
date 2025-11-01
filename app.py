import streamlit as st
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite
import os

# ---------------- CONFIG ----------------
MODEL_PATH = "MobileNetV2_model.tflite"
INPUT_SIZE = (224, 224)
LABELS = ["Normal", "Cataract", "Glaucoma", "Diabetic Retinopathy"]
# ----------------------------------------

st.set_page_config(page_title="Retinal Disease Classifier", layout="centered")
st.title("🩺 Retinal Disease Classifier")
st.write("Upload a retinal image to detect possible diseases using MobileNetV2 (TFLite).")

# ---- Load Model ----
@st.cache_resource
def load_model(path):
    interpreter = tflite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model(MODEL_PATH)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ---- Preprocess ----
def preprocess_image(image):
    image = image.convert("RGB").resize(INPUT_SIZE)
    img = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(img, axis=0)

# ---- Predict ----
def predict(img):
    input_data = preprocess_image(img)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return np.squeeze(output_data)

uploaded = st.file_uploader("📤 Upload retinal image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Predict"):
        preds = predict(image)
        exp = np.exp(preds - np.max(preds))
        probs = exp / np.sum(exp)
        top = np.argmax(probs)

        st.success(f"### 🧠 Prediction: {LABELS[top]}")
        st.progress(float(probs[top]))
        st.info(f"Confidence: {probs[top]*100:.2f}%")
