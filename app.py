import streamlit as st
from PIL import Image

# -----------------------------------------
# Title and Description
# -----------------------------------------
st.set_page_config(page_title="Retinal Disease Classifier", page_icon="👁️", layout="centered")

st.title("👁️ Retinal Disease Classification")
st.markdown("""
Upload a **retinal image** to get a disease prediction.  
This is a **demo app** (model integration coming soon).
""")

# -----------------------------------------
# Image Upload
# -----------------------------------------
uploaded_file = st.file_uploader("📤 Upload a retinal image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Retinal Image", use_column_width=True)
    
    # -----------------------------------------
    # Demo Prediction Output
    # -----------------------------------------
    st.success("✅ Predicted Disease: **Normal (Demo)**")
    st.progress(85)
    st.caption("Confidence: 85% (Demo)")

    st.markdown("""
    ### 🧠 Model Insights (Demo)
    - The model detected clear and healthy optic disc.
    - No visible signs of diabetic retinopathy or glaucoma.
    - Consistent with normal retinal structure.
    """)

else:
    st.info("Please upload a retinal image to start the demo.")

# -----------------------------------------
# Footer
# -----------------------------------------
st.markdown("""
---
**Made by Uroosha Usman**  
_MSc Computer Science | Lucknow University_
""")
