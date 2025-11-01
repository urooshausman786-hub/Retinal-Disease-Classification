import streamlit as st
from PIL import Image
import numpy as np
import tflite_runtime.interpreter as tflite

st.title("🩺 Retinal Disease Classification App")
st.write("Upload a retinal image to classify the disease using MobileNetV2 (TFLite Model).")

# Load labels
with open("labels.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Load TFLite model
@st.cache_resource
def load_model():
    interpreter = tflite.Interpreter(model_path="MobileNetV2_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Preprocess function
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Predict function
def predict(image):
    input_data = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# Upload image
uploaded_file = st.file_uploader("📤 Upload a retinal image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("🔍 Classify Image"):
        preds = predict(image)
        class_index = np.argmax(preds)
        confidence = np.max(preds)
        st.success(f"**Prediction:** {class_names[class_index]}")
        st.info(f"**Confidence:** {confidence * 100:.2f}%")
        st.caption("✅ Model running on TFLite Runtime (no TensorFlow dependency)")
