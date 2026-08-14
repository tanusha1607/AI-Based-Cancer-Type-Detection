import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="AI Cancer Detector",
    page_icon="🧬",
    layout="wide"
)

# ---------- ADVANCED CSS ----------
st.markdown("""
<style>

/* background */
.stApp {
    background: radial-gradient(circle at top left, #0f172a, #020617);
}

/* animated title */
.title {
    text-align: center;
    font-size: 48px;
    font-weight: 800;
    background: linear-gradient(90deg, #ff4b4b, #8b5cf6, #22c55e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: glow 3s ease-in-out infinite alternate;
}

@keyframes glow {
    from {filter: drop-shadow(0 0 5px #8b5cf6);}
    to {filter: drop-shadow(0 0 20px #22c55e);}
}

/* subtitle */
.subtitle {
    text-align: center;
    color: #cbd5e1;
    margin-bottom: 25px;
}

/* glass card */
.glass {
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(14px);
    border-radius: 16px;
    padding: 20px;
    border: 1px solid rgba(255,255,255,0.08);
}

/* prediction highlight */
.top-pred {
    font-size: 22px;
    font-weight: 700;
    color: #22c55e;
}

/* confidence badge */
.conf-badge {
    background: linear-gradient(90deg,#22c55e,#4ade80);
    padding: 6px 14px;
    border-radius: 999px;
    color: black;
    font-weight: 700;
    display: inline-block;
    margin-top: 6px;
}

</style>
""", unsafe_allow_html=True)

# ---------- LOAD MODEL ----------
model = tf.keras.models.load_model('Model.h5')

labels = [
    'Colon Adenocarcinoma',
    'Colon Benign Tissue',
    'Lung Adenocarcinoma',
    'Lung Benign Tissue',
    'Lung Squamous Cell Carcinoma'
]

# ---------- FUNCTIONS ----------
def preprocess_image(uploaded_file):
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_cancer(img_array):
    predictions = model.predict(img_array)[0]
    return {labels[i]: float(predictions[i]) for i in range(len(labels))}

def plot_pie(predictions):
    fig, ax = plt.subplots()
    ax.pie(predictions.values(), labels=predictions.keys(),
           autopct='%1.1f%%', startangle=140)
    ax.axis('equal')
    st.pyplot(fig)

# ---------- HEADER ----------
st.markdown('<div class="title">🧬 AI Cancer Type Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Deep Learning Powered Histopathology Analysis</div>', unsafe_allow_html=True)

# ---------- UPLOADER ----------
uploaded_file = st.file_uploader(
    "📤 Upload Medical Image",
    type=["jpg", "png", "jpeg"]
)

# ---------- MAIN ----------
if uploaded_file is not None:

    img_array = preprocess_image(uploaded_file)
    predictions = predict_cancer(img_array)

    top_label = max(predictions, key=predictions.get)
    top_prob = predictions[top_label] * 100

    col1, col2 = st.columns([1.1, 1])

    # LEFT CARD
    with col1:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.image(uploaded_file, caption="Uploaded Histopathology Image", width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    # RIGHT CARD
    with col2:
        st.markdown('<div class="glass">', unsafe_allow_html=True)

        st.markdown(
            f'<div class="top-pred">🎯 {top_label}</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="conf-badge">Confidence: {top_prob:.2f}%</div>',
            unsafe_allow_html=True
        )

        st.markdown("### 📊 Class Probabilities")

        for label, prob in predictions.items():
            st.write(label)
            st.progress(float(prob))
            st.caption(f"{prob*100:.2f}%")

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### 🥧 Prediction Distribution")
    plot_pie(predictions)