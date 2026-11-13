import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from matplotlib import colormaps
from PIL import Image
 
st.set_page_config(page_title="SkinScan AI", page_icon="🩺", layout="centered")
THRESHOLD = 0.3
 
CSS = """
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
div[data-baseweb="tab-list"] { gap: 8px; background: #E4ECEF; padding: 6px; border-radius: 14px; }
div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] { display: none; }
button[data-baseweb="tab"] { flex: 1; justify-content: center; height: 44px; border-radius: 10px;
        background: transparent; font-weight: 500; transition: all .2s ease; }
button[data-baseweb="tab"]:hover { background: rgba(14,124,134,.12); }
button[data-baseweb="tab"][aria-selected="true"] { background: #0E7C86;
        box-shadow: 0 4px 12px rgba(14,124,134,.35); }
button[data-baseweb="tab"][aria-selected="true"] p { color: #fff; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)
 
 
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
 
 
HERO = (
    '<div class="hero">'
    "<h1>SkinScan AI</h1>"
    "<p>Upload a close-up photo of a skin lesion to get an AI screening result, "
    "a heatmap of what the model focused on, and performance insights.</p>"
    "</div>"
)
st.markdown(HERO, unsafe_allow_html=True)
 
file = st.file_uploader(
    "Upload a skin lesion image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)
 
img = None
p = None
cam = None
if file:
    img = Image.open(file).convert("RGB")
    with st.spinner("Analyzing image..."):
        arr, p = predict(img)
        cam = gradcam(arr)
 
tab1, tab2, tab3, tab4 = st.tabs(["Screening", "Heatmap", "Model insights", "About"])
 
with tab1:
    if img is None:
        st.markdown(
            '<div class="card"><p class="verdict">Ready when you are</p>'
            '<p class="sub" style="margin:0">Upload a clear, well-lit JPG or PNG to begin.</p></div>',
            unsafe_allow_html=True,
        )
    else:
        st.image(img, use_container_width=True)
        malignant = p > THRESHOLD
        color = "#C2410C" if malignant else "#1F8A5B"
        verdict = "Likely malignant" if malignant else "Likely benign"
        if malignant:
            advice = "This lesion shows features linked to malignancy. Please consult a dermatologist."
        else:
            advice = "No strong signs of malignancy were detected. Keep monitoring any changes."
 
        card = (
            '<div class="card">'
            f'<p class="verdict" style="color:{color}">{verdict}</p>'
            f'<p class="sub">{advice}</p>'
            '<div class="track">'
            f'<div class="fill" style="width:{p * 100:.1f}%; background:{color}"></div>'
            f'<div class="mark" style="left:{THRESHOLD * 100:.0f}%"></div>'
            "</div>"
            '<div class="scale"><span>0%</span>'
            f"<span>Malignancy score: {p:.1%}</span><span>100%</span></div>"
            f'<p class="note">The dark line marks the alert threshold ({THRESHOLD:.0%}). '
            "For education and screening support only, not a medical diagnosis.</p>"
            "</div>"
        )
        st.markdown(card, unsafe_allow_html=True)
 
        st.markdown("##### Class probabilities")
        st.bar_chart(
            pd.DataFrame({"Probability": [1 - p, p]}, index=["Benign", "Malignant"]),
            horizontal=True,
        )
 
with tab2:
    if img is None:
        st.info("Upload an image above to see the heatmap.")
    else:
        heat_img, mix = overlay(img, cam)
        c1, c2, c3 = st.columns(3)
        c1.image(img, caption="Original", use_container_width=True)
        c2.image(heat_img, caption="Heatmap", use_container_width=True)
        c3.image(mix, caption="Overlay", use_container_width=True)
        st.caption(
            "Warm colors (red, yellow) show regions that pushed the model most toward "
            "'malignant'. Heatmaps explain model attention, not a diagnosis, and can "
            "sometimes highlight skin or background instead of the lesion."
        )
 
with tab3:
    m1, m2, m3 = st.columns(3)
    m1.metric("Test accuracy", "90%")
    m2.metric("Malignant recall", "92%", help="At threshold 0.3")
    m3.metric("Test images", "1,000")
    st.markdown("##### Threshold trade-off (test set)")
    df = pd.DataFrame(
        {
            "Threshold": ["0.50", "0.40", "0.30", "0.25"],
            "Malignant recall": [0.88, 0.90, 0.92, 0.92],
            "Benign recall": [0.92, 0.91, 0.87, 0.85],
        }
    ).set_index("Threshold")
    st.line_chart(df)
    st.caption(
        "Lowering the threshold catches more malignant cases but raises false alarms. "
        "This app uses 0.30."
    )
 
with tab4:
    st.markdown("**How it works**")
    st.write(
        "A pretrained EfficientNetB0 network was fine-tuned on about 9,600 labeled images "
        "of benign and malignant skin lesions, then tested on 1,000 separate images."
    )
    st.markdown("**Limits**")
    st.write(
        "The model was trained on one public dataset. Photos from other cameras, lighting "
        "or skin tones may give weaker results."
    )
    st.markdown("**Disclaimer**")
    st.write(
        "SkinScan AI is a learning project. It is not a medical device and cannot replace "
        "a dermatologist."
    )
