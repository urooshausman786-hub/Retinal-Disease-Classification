import streamlit as st
import numpy as np
from PIL import Image
import io
import os
import traceback
import tensorflow as tf  # ✅ use TensorFlow directly (includes TFLite interpreter)

# ---------------- CONFIG ----------------
MODEL_PATH = "MobileNetV2_model.tflite"  # your model file
INPUT_SIZE = (224, 224)                  # input shape
LABELS = ["Normal", "Cataract", "Glaucoma", "Diabetic Retinopathy"]  # update to your real classes
# ----------------------------------------

st.set_page_config(page_title="Retinal Classifier", layout="centered")
st.title("🩺 Retinal Disease Classifier")
st.write("Upload a retinal image — model predicts the disease using MobileNetV2 (TFLite).")

# ---- Load TFLite Model ----
@st.cache_resource
def load_model(model_path):
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        st.stop()
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

try:
    interpreter, input_details, output_details = load_model(MODEL_PATH)
    st.success("✅ Model loaded successfully.")
except Exception as e:
    st.error("Model loading failed.")
    st.text(traceback.format_exc())
    st.stop()

# ---- Helper: Preprocess Image ----
def preprocess_image(image):
    image = image.convert("RGB").resize(INPUT_SIZE)
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ---- Helper: Predict ----
def predict_image(image):
    input_data = preprocess_image(image)
    input_index = interpreter.get_input_details()[0]['index']
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
    return np.squeeze(output_data)

# ---- File Upload ----
uploaded_file = st.file_uploader("📤 Upload Retinal Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="🩸 Uploaded Image", use_column_width=True)

    if st.button("🔍 Classify Disease"):
        with st.spinner("Analyzing... please wait ⏳"):
            try:
                preds = predict_image(image)
                exp = np.exp(preds - np.max(preds))
                probs = exp / np.sum(exp)
                top_index = np.argmax(probs)
                predicted_class = LABELS[top_index] if top_index < len(LABELS) else f"Class_{top_index}"
                confidence = probs[top_index] * 100

                st.success(f"### 🧠 Prediction: {predicted_class}")
                st.info(f"**Confidence:** {confidence:.2f}%")

                st.write("### 📊 Class Probabilities:")
                st.dataframe({
                    "Class": LABELS,
                    "Confidence (%)": [round(float(p) * 100, 2) for p in probs]
                })
            except Exception as e:
                st.error("⚠️ Error during prediction.")
                st.text(traceback.format_exc())
else:
    st.info("⬆️ Upload an image to start diagnosis.")
    st.caption("Ensure your `.tflite` model file is in the same folder as this app.")
