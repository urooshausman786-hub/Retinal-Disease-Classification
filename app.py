
import tflite_runtime.interpreter as tflite
import numpy as np
from PIL import Image
import streamlit as st

# --- CONFIG ---
MODEL_PATH = "MobileNetV2_model.tflite"
INPUT_SIZE = (224, 224)
LABELS = ["Class0", "Class1", "Class2", "Class3"]  # replace with your class names
# ---------------

st.set_page_config(page_title="Retinal Disease Classifier", layout="centered")
st.title("🩺 Retinal Disease Classifier (TensorFlow Lite)")
st.write("Upload a retinal image to get model predictions.")

@st.cache_resource
def load_interpreter(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter, interpreter.get_input_details(), interpreter.get_output_details()

def preprocess_image(image: Image.Image, input_shape):
    img = image.convert("RGB").resize((input_shape[1], input_shape[2]))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict(interpreter, input_details, output_details, data):
    input_index = input_details[0]['index']
    output_index = output_details[0]['index']
    interpreter.set_tensor(input_index, data.astype(input_details[0]['dtype']))
    interpreter.invoke()
    output = interpreter.get_tensor(output_index)
    return np.squeeze(output)

try:
    interpreter, input_details, output_details = load_interpreter(MODEL_PATH)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error("❌ Failed to load model: " + str(e))
    st.text(traceback.format_exc())
    st.stop()

uploaded = st.file_uploader("Upload retinal image (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(io.BytesIO(uploaded.read()))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    inp_shape = input_details[0]["shape"]
    if inp_shape[0] == 0:
        inp_shape[0] = 1
    pre = preprocess_image(image, inp_shape)

    try:
        preds = predict(interpreter, input_details, output_details, pre)
        exp = np.exp(preds - np.max(preds))
        probs = exp / np.sum(exp)
        top_k = probs.argsort()[-3:][::-1]
        st.subheader("Top Predictions:")
        for i in top_k:
            label = LABELS[i] if i < len(LABELS) else f"Class_{i}"
            st.write(f"**{label}** — {probs[i]:.4f}")
    except Exception as e:
        st.error("Inference failed.")
        st.text(traceback.format_exc())
else:
    st.info("Please upload an image to start.")

