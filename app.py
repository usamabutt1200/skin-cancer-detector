import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Skin Cancer Detector", page_icon="🩺")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("skin_cancer_final.keras")

model = load_model()
THRESHOLD = 0.3

st.title("🩺 Skin Cancer Detector")
st.write("Skin lesion ki photo upload karein.")

file = st.file_uploader("Image chunein", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, use_container_width=True)

    arr = np.array(img.resize((224, 224)), dtype=np.float32)[None, ...]
    p = float(model.predict(arr)[0][0])

    if p > THRESHOLD:
        st.error(f"🔴 Malignant ka imkan: {p:.1%}")
    else:
        st.success(f"🟢 Benign ka imkan: {1 - p:.1%}")

    st.caption("⚠️ Yeh sirf seekhne ke liye hai, tibbi tashkhees nahi. Doctor se rujoo karein.")