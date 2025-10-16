import streamlit as st
from PIL import Image
import numpy as np
import pickle

# Load your trained model (replace with your actual model file)
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Retinal Disease Classification")
st.write("Upload a retinal image to classify the disease.")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess image according to your model (example: resize and normalize)
    image = image.resize((224, 224))
    image_array = np.array(image)/255.0
    image_array = image_array.reshape(1, 224, 224, 3)  # adjust shape to match your model
    
    # Predict
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction, axis=1)
    
    # Map predicted class to label (adjust according to your dataset)
    disease_dict = {0: "Normal", 1: "Diabetic Retinopathy", 2: "Glaucoma", 3: "Macular Degeneration"}
    st.write(f"Predicted Disease: **{disease_dict[predicted_class[0]]}**")
