import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
tflite = tf.lite

st.set_page_config(page_title="Retinal Disease Classification", page_icon="👁️", layout="wide")

st.title("👁️ Retinal Disease Classification using MobileNetV2 (TFLite)")
st.markdown("Upload a retinal image to classify it into disease categories.")

# Load model
@st.cache_resource
def load_model():
    interpreter = tflite.Interpreter(model_path="MobileNetV2_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define class names (update as per your dataset)
CLASS_NAMES = [
    "Normal",
    "Diabetic Retinopathy",
    "Glaucoma",
    "Cataract",
    "Age-related Macular Degeneration"
]

# File uploader
uploaded_file = st.file_uploader("📤 Upload a retinal image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Preprocess image
    img = image.resize((224, 224))  # Adjust based on your MobileNetV2 input size
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]

    # Get predicted class
    predicted_index = np.argmax(predictions)
    confidence = predictions[predicted_index]

    st.success(f"### 🩺 Prediction: {CLASS_NAMES[predicted_index]}")
    st.info(f"Confidence: {confidence * 100:.2f}%")

    # Show probability chart
    st.bar_chart(predictions)

st.markdown("---")
st.caption("Developed by Uroosha Usman | MSc Computer Science | Lucknow University 💻")
