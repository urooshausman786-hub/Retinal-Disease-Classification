
import streamlit as st
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

# Load the model
@st.cache_resource
def load_model():
    interpreter = tflite.Interpreter(model_path="retinal_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

# Get input & output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Streamlit UI
st.title("🩺 Retinal Disease Classification")
st.write("Upload a retinal image to classify the disease using a CNN model.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image
    img = image.resize((224, 224))  # adjust to your model's input size
    img_array = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = np.argmax(output_data)

    # Class labels (update these as per your model)
    labels = ["Normal", "Diabetic Retinopathy", "Glaucoma", "Cataract", "Age-related Macular Degeneration"]

    st.subheader("🔍 Prediction:")
    st.success(f"**{labels[prediction]}**")

    st.caption("Model: Retinal Disease Classifier (TFLite)")
