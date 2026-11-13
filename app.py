import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from matplotlib import colormaps
from PIL import Image

st.set_page_config(page_title="SkinScan AI", page_icon="🩺", layout="centered")
THRESHOLD = 0.3

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');
html, body, [class*="css"], .stApp { font-family: 'DM Sans', sans-serif; }
.stApp { background: #F3F7F8; color: #0F2A3D; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { max-width: 820px; padding-top: 2rem; }
.hero { background: linear-gradient(135deg, #0F2A3D 0%, #0E7C86 100%);
        border-radius: 20px; padding: 2rem; color: #fff; margin-bottom: 1.2rem; }
.hero h1 { font-size: 2.1rem; font-weight: 700; margin: 0 0 .4rem 0; color: #fff; }
.hero p { margin: 0; opacity: .88; line-height: 1.5; }
.card { background: #fff; border: 1px solid #DCE6E9; border-radius: 16px;
        padding: 1.4rem 1.5rem; margin-top: 1rem; }
.verdict { font-size: 1.5rem; font-weight: 700; margin: 0 0 .2rem 0; }
.sub { color: #5B7080; margin: 0 0 1.1rem 0; }
.track { position: relative; height: 12px; background: #E4ECEF; border-radius: 99px; }
.fill { height: 12px; border-radius: 99px; }
.mark { position: absolute; top: -5px; width: 2px; height: 22px; background: #0F2A3D; }
.scale { display: flex; justify-content: space-between; font-size: .8rem;
         color: #5B7080; margin-top: .45rem; }
.note { font-size: .85rem; color: #5B7080; margin-top: 1.2rem; line-height: 1.5; }
[data-testid="stFileUploader"] section { background: #fff; border: 2px dashed #9DB9C2;
        border-radius: 16px; }
[data-testid="stImage"] img { border-radius: 14px; }
button[data-baseweb="tab"] { font-weight: 500; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("skin_cancer_final.keras")


model = load_model()


def predict(img):
    arr = np.array(img.resize((224, 224)), dtype=np.float32)[None, ...]
    return arr, float(model.predict(arr, verbose=0)[0][0])


def gradcam(arr):
    base = model.get_layer("efficientnetb0")
    head = model.get_layer("dense")
    x = tf.convert_to_tensor(arr)
    with tf.GradientTape() as tape:
        conv = base(x, training=False)
        tape.watch(conv)
        score = head(tf.reduce_mean(conv, axis=(1, 2)))
    grads = tape.gradient(score, conv)
    w = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.nn.relu(tf.reduce_sum(conv * w[:, None, None, :], axis=-1))[0].numpy()
    return cam / (cam.max() + 1e-8)


def overlay(img, cam, alpha=0.45):
    heat = Image.fromarray(np.uint8(cam * 255)).resize(img.size, Image.BICUBIC)
    rgb = (colormaps["jet"](np.array(heat) / 255.0)[..., :3] * 255).astype(np.uint8)
    heat_img = Image.fromarray(rgb)
    return heat_img, Image.blend(img, heat_img, alpha)


st.markdown("""
<div class="hero">
  <h1>SkinScan AI</h1>
  <p>Upload a close-up photo of a skin
