import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="SkinScan AI", page_icon="🩺", layout="centered")

THRESHOLD = 0.3

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');
html, body, [class*="css"], .stApp { font-family: 'DM Sans', sans-serif; }
.stApp { background: #F3F7F8; color: #0F2A3D; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { max-width: 760px; padding-top: 2.5rem; }

.hero { background: linear-gradient(135deg, #0F2A3D 0%, #0E7C86 100%);
        border-radius: 20px; padding: 2.2rem 2rem; color: #fff; margin-bottom: 1.5rem; }
.hero h1 { font-size: 2.1rem; font-weight: 700; margin: 0 0 .4rem 0; color: #fff; }
.hero p { margin: 0; font-size: 1.02rem; opacity: .88; line-height: 1.5; }

.card { background: #fff; border: 1px solid #DCE6E9; border-radius: 16px;
        padding: 1.4rem 1.5rem; margin-top: 1rem; }
.verdict { font-size: 1.5rem; font-weight: 700; margin: 0 0 .2rem 0; }
.sub { color: #5B7080; margin: 0 0 1.1rem 0; }
.track { position: relative; height: 12px; background: #E4ECEF; border-radius: 99px; }
.fill { height: 12px; border-radius: 99px; }
.mark { position: absolute; top: -5px; width: 2px; height: 22px; background: #0F2A3D; }
.scale { display: flex; justify-content: space-between; font-size: .8rem;
         color: #5B7080; margin-top: .45rem; }
.note { font-size: .85rem; color: #5B7080; margin-top: 1.4rem; line-height: 1.5; }

[data-testid="stFileUploader"] section { background: #fff; border: 2px dashed #9DB9C2;
        border-radius: 16px; padding: 1.4rem; }
[data-testid="stImage"] img { border-radius: 14px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("skin_cancer_final.keras")


model = load_model()

st.markdown("""
<div class="hero">
  <h1>SkinScan AI</h1>
  <p>Upload a close-up photo of a skin lesion and get an instant AI screening result.</p>
</div>
""", unsafe_allow_html=True)

file = st.file_uploader("Upload a skin lesion image", type=["jpg", "jpeg", "png"],
                        label_visibility="collapsed")

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, use_container_width=True)

    with st.spinner("Analyzing image..."):
        arr = np.array(img.resize((224, 224)), dtype=np.float32)[None, ...]
        p = float(model.predict(arr, verbose=0)[0][0])

    malignant = p > THRESHOLD
    color = "#C2410C" if malignant else "#1F8A5B"
    verdict = "Likely malignant" if malignant else "Likely benign"
    advice = ("This lesion shows features linked to malignancy. Please consult a dermatologist."
              if malignant else
              "No strong signs of malignancy were detected. Keep monitoring any changes.")

    st.markdown(f"""
    <div class="card">
      <p class="verdict" style="color:{color}">{verdict}</p>
      <p class="sub">{advice}</p>
      <div class="track">
        <div class="fill" style="width:{p*100:.1f}%; background:{color}"></div>
        <div class="mark" style="left:{THRESHOLD*100:.0f}%"></div>
      </div>
      <div class="scale"><span>0%</span>
        <span>Malignancy score: {p:.1%}</span><span>100%</span></div>
      <p class="note">The dark line marks the alert threshold ({THRESHOLD:.0%}).
      This tool is for education and screening support only. It is not a medical diagnosis.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="card">
      <p class="verdict">Ready when you are</p>
      <p class="sub" style="margin:0">Upload a clear, well-lit JPG or PNG to begin.</p>
    </div>
    """, unsafe_allow_html=True)
