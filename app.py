import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="🩺 Retinal Disease Classification",
    page_icon="👁️",
    layout="wide"
)

# -----------------------------
# Custom CSS for purple theme
# -----------------------------
st.markdown("""
<style>
/* Bright gradient background */
.stApp {
    background: linear-gradient(to bottom, #ffffff, #f3e8ff);
    color: #1e293b;
    max-width: 900px;
    margin: auto;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    padding: 20px 40px;
}

/* Header */
.custom-header {
    color: #7c3aed;
    text-align: center;
    font-size: 50px;
    font-weight: 700;
    margin-bottom: 5px;
}

/* Subtitle */
.custom-subtitle {
    color: #d8b4fe;
    text-align: center;
    font-size: 22px;
    font-weight: 500;
    margin-top: 0;
    margin-bottom: 30px;
}

/* File uploader */
.css-1v0mbdj.edgvbvh3 {
    border: 2px dashed #c084fc;
    border-radius: 12px;
    padding: 25px;
    background-color: rgba(255, 255, 255, 0.9);
    color: #1e293b;
}

/* Image display */
.stImage {
    border-radius: 15px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.08);
}

/* Prediction card */
.prediction-card {
    background: linear-gradient(to right, #e0c3fc, #8ec5fc);
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    text-align: center;
    font-size: 24px;
    font-weight: 600;
    color: #1e3a8a;
    margin-top: 20px;
}

/* Info box */
.stInfo {
    font-size: 18px;
    background-color: #ede9fe !important;
    color: #7c3aed !important;
    border-radius: 8px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header & Subtitle
# -----------------------------
st.markdown('<div class="custom-header">🩺 Retinal Disease Classification System</div>', unsafe_allow_html=True)

# -----------------------------
# File uploader
# -----------------------------
uploaded_file = st.file_uploader(
    "📤 Drag & drop a retinal image here or click to browse (JPG, PNG, max 200MB)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    # Open image
    image = Image.open(uploaded_file).convert("RGB")

    # Resize image proportionally if too large
    max_display_width = 300
    w, h = image.size
    if w > max_display_width:
        new_height = int(h * (max_display_width / w))
        image = image.resize((max_display_width, new_height))

    # Display uploaded image
    st.image(image, caption="Uploaded Retinal Image", use_column_width=False)

    st.info("👁️ Running model inference...")

    # -----------------------------
    # Load TFLite model
    # -----------------------------
    interpreter = tf.lite.Interpreter(model_path="MobileNetV2_model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preprocess image for model
    img = image.resize((224, 224))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    # Predicted class and confidence
    classes = ["Diabetic Retinopathy", "Glaucoma", "Cataract", "Normal"]
    pred_class = np.argmax(output_data)
    confidence = output_data[pred_class]

    # Display results in a purple gradient card
    st.markdown(f"""
        <div class="prediction-card">
            ✅ Predicted Disease: <b>{classes[pred_class]}</b><br>
            Confidence: {confidence*100:.2f}%
        </div>
    """, unsafe_allow_html=True)

else:
    st.info("👁️ Please upload a retinal image to begin diagnosis.")
