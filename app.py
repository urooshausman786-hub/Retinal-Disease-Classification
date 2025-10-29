import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

# -----------------------------
# Load your trained model
# -----------------------------
@st.cache_resource
def load_retinal_model():
    model = load_model("retinal_model.h5")   # make sure this file is in same folder
    return model

model = load_retinal_model()

# -----------------------------
# Define class labels (update if different)
# -----------------------------
CLASS_NAMES = ["Normal", "Diabetic Retinopathy", "Glaucoma", "Cataract", "Age-related Macular Degeneration"]

# -----------------------------
# Streamlit App UI
# -----------------------------
st.title("👁️ Retinal Disease Classification")
st.write("Upload a retinal image to get an AI-based disease prediction.")

# Upload image
uploaded_file = st.file_uploader("Choose a retinal image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Retinal Image', use_column_width=True)
    
    # Preprocess image for model
    img = image.resize((224, 224))  # adjust size based on your model’s input
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0   # normalization (same as training)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)

    # Show results
    st.success(f"Predicted Disease: **{CLASS_NAMES[predicted_class]}**")
    st.write(f"Model Confidence: **{confidence*100:.2f}%**")

    # Optionally show probability for each class
    st.subheader("Prediction Probabilities:")
    for i, class_name in enumerate(CLASS_NAMES):
        st.write(f"{class_name}: {prediction[0][i]*100:.2f}%")

else:
    st.info("Please upload an image to start prediction.")


