import streamlit as st
from PIL import Image
import numpy as np

# Try importing tflite_runtime first; if not found, fall back to tensorflow
try:
    import tflite_runtime.interpreter as tflite
    USE_TFLITE = True
except ModuleNotFoundError:
    import tensorflow as tf
    USE_TFLITE = False

st.title("🩺 Retinal Disease Classification")

# Load class labels
with open("labels.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Load model
@st.cache_resource
def load_model():
    if USE_TFLITE:
        interpreter = tflite.Interpreter(model_path="MobileNetV2_model.tflite")
        interpreter.allocate_tensors()
        return interpreter
    else:
        model = tf.lite.Interpreter(model_path="MobileNetV2_model.tflite")
        model.allocate_tensors()
        return model

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Preprocessing
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image, dtype=np.float32)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Prediction
def predict(image):
    input_data = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# File upload
uploaded_file = st.file_uploader("Upload a retinal image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("🔍 Predict Disease"):
        prediction = predict(image)
        class_index = np.argmax(prediction)
        confidence = np.max(prediction)
        st.success(f"Predicted: **{class_names[class_index]}**")
        st.info(f"Confidence: {confidence*100:.2f}%")

        if USE_TFLITE:
            st.caption("Running on TFLite Runtime ✅")
        else:
            st.caption("Running on TensorFlow (fallback mode) ⚙️")

