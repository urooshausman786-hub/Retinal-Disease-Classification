# app.py
import streamlit as st
import numpy as np
from PIL import Image
import io
import os
import sys
import traceback

st.set_page_config(page_title="Retinal Classifier", layout="centered")

# --- CONFIG: change if needed ---
MODEL_PATH = "MobileNetV2_model.tflite"   # put your .tflite here
INPUT_SIZE = (224, 224)                   # expected by your model
LABELS = ["Class0", "Class1", "Class2", "Class3"]  # replace with your actual class names (len must match)
# ----------------------------------

st.title("🩺 Retinal Disease Classifier (TFLite)")
st.write("Upload a retinal image — app runs a TFLite model and shows top predictions.")

# Try to import tflite_runtime first (smaller), otherwise fallback to tensorflow
use_tflite_runtime = False
interpreter_module = None
try:
    import tflite_runtime.interpreter as tflite  # type: ignore
    interpreter_module = tflite
    use_tflite_runtime = True
    st.write("Using `tflite_runtime` for inference.")
except Exception:
    try:
        import tensorflow as tf
        interpreter_module = tf.lite
        st.write("`tflite_runtime` not available — falling back to `tensorflow`'s tflite interpreter.")
    except Exception as e:
        st.error("ERROR: Neither tflite_runtime nor tensorflow available in environment.")
        st.stop()

@st.cache_resource
def load_interpreter(model_path: str):
    """Load and return TFLite interpreter and input/output details."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    try:
        if use_tflite_runtime:
            interpreter = interpreter_module.Interpreter(model_path=model_path)
        else:
            # tensorflow.lite
            import tensorflow as tf  # re-import inside to be safe
            interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        return interpreter, input_details, output_details
    except Exception as e:
        st.error("Failed to load TFLite interpreter.")
        st.text(traceback.format_exc())
        raise e

def preprocess_image(image: Image.Image, input_shape):
    """Resize, normalize and return a float32 np array shaped for interpreter"""
    img = image.convert("RGB").resize((input_shape[1], input_shape[2]))
    arr = np.asarray(img).astype(np.float32)
    # Common normalization for models trained with ImageNet: scale to [0,1]
    arr = arr / 255.0
    # If model expects [-1,1] uncomment below and comment the /255.0 above:
    # arr = (arr / 127.5) - 1.0
    # Expand dims to [1, H, W, C]
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_with_interpreter(interpreter, input_details, output_details, preprocessed):
    """Run inference and return raw output"""
    # Ensure dtype matches
    input_index = input_details[0]['index']
    # If model expects int8 quantized inputs, we must scale & cast
    if input_details[0]['dtype'] == np.uint8 or input_details[0]['dtype'] == np.int8:
        # quantization params
        scale, zero_point = input_details[0].get('quantization', (1.0, 0))
        if scale == 0:
            scale = 1.0
        preprocessed_q = (preprocessed / scale + zero_point).astype(input_details[0]['dtype'])
        interpreter.set_tensor(input_index, preprocessed_q)
    else:
        interpreter.set_tensor(input_index, preprocessed.astype(input_details[0]['dtype']))

    interpreter.invoke()

    output_index = output_details[0]['index']
    output_data = interpreter.get_tensor(output_index)

    # If output is quantized, dequantize
    if output_details[0]['dtype'] in (np.uint8, np.int8):
        out_scale, out_zero = output_details[0].get('quantization', (1.0, 0))
        if out_scale != 0:
            output_data = (output_data.astype(np.float32) - out_zero) * out_scale

    return output_data

# Load interpreter (show helpful debug)
try:
    interpreter, input_details, output_details = load_interpreter(MODEL_PATH)
    st.write("Model loaded. Input details:")
    st.text(str(input_details))
    st.text("Output details:")
    st.text(str(output_details))
except Exception as e:
    st.error(str(e))
    st.stop()

# UI: upload
uploaded = st.file_uploader("Upload retinal image (jpg/png)", type=['jpg','jpeg','png'])

# Option: show example image button (if you add one to repo)
if uploaded:
    img = Image.open(io.BytesIO(uploaded.read()))
    st.image(img, caption="Uploaded image", use_column_width=True)

    # Preprocess
    # Input details can be e.g., {'shape':[1,224,224,3], ...}
    inp_shape = input_details[0]['shape']
    # If the interpreter's shape has 0 for batch, replace with 1
    if inp_shape[0] == 0:
        inp_shape[0] = 1
    try:
        pre = preprocess_image(img, inp_shape)
    except Exception as e:
        st.error("Preprocessing failed: " + str(e))
        st.stop()

    # Predict
    try:
        out = predict_with_interpreter(interpreter, input_details, output_details, pre)
    except Exception as e:
        st.error("Model inference failed.")
        st.text(traceback.format_exc())
        st.stop()

    # Postprocess
    # If model outputs logits or probabilities, handle accordingly.
    probs = np.squeeze(out)
    # If model output is single logit per class, apply softmax
    if probs.ndim == 1:
        try:
            # softmax with numeric stability
            exp = np.exp(probs - np.max(probs))
            probs = exp / np.sum(exp)
        except Exception:
            pass

    # Top-k results
    top_k = min(3, len(probs))
    top_idx = probs.argsort()[-top_k:][::-1]
    st.subheader("Top predictions")
    for i in top_idx:
        label = LABELS[i] if i < len(LABELS) else f"Class_{i}"
        st.write(f"**{label}** — {probs[i]:.4f}")

    st.write("---")
    st.write("Full probabilities:")
    st.dataframe({(LABELS[i] if i < len(LABELS) else f"Class_{i}"): float(probs[i]) for i in range(len(probs))})
else:
    st.info("Upload an image to run inference. Make sure your TFLite model file is present as: " + MODEL_PATH)
    st.markdown("<small>Tip: change LABELS in app.py to match your dataset class names.</small>", unsafe_allow_html=True)
