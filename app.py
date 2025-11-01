import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("👁️ Retinal Disease Classification (MobileNetV2)")

# Load TFLite model
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="MobileNetV2_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

# Get input & output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# File uploader
uploaded_file = st.file_uploader("Upload a retinal image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).resize((224, 224))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0]

    classes = ["Cataract", "Diabetic Retinopathy", "Glaucoma", "Normal", "Other"]
    result = classes[np.argmax(preds)]
    st.success(f"🩺 Prediction: **{result}**")

