import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image

# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="Oral Disease Classifier",
    page_icon="🦷",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ============================================================
# Custom CSS for a polished look
# ============================================================
st.markdown("""
    <style>
    .main {
        background-color: #FAFAFA;
    }
    .stButton>button {
        background-color: #2E86AB;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
    }
    .result-box {
        padding: 1.2rem;
        border-radius: 12px;
        background-color: #F0F7FF;
        border-left: 5px solid #2E86AB;
        margin-top: 1rem;
        color: #1B1B1B;
    }
    .result-box h3, .result-box p {
        color: #1B1B1B !important;
    }
    .result-box h2 {
        color: #2E86AB !important;
    }
    h1 {
        color: #1B4965;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# Load Model (cached so it only loads once)
# ============================================================
@st.cache_resource
def load_model():
    return keras.models.load_model("final_oral_disease_model.keras")

model = load_model()

CLASS_NAMES = ["Calculus", "Caries", "Gingivitis", "Hypodontia", "Mouth Ulcer", "Tooth Discoloration"]
CLASS_INFO = {
    "Calculus": "Hardened plaque deposits (tartar) on the teeth.",
    "Caries": "Tooth decay caused by erosion of the enamel.",
    "Gingivitis": "Gum inflammation, usually caused by bacterial plaque buildup.",
    "Hypodontia": "Congenital absence of one or more teeth.",
    "Mouth Ulcer": "A sore or lesion on the soft tissue of the mouth.",
    "Tooth Discoloration": "A change in the natural color of the teeth.",
}

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.header("ℹ️ About This Project")
    st.write(
        "An EfficientNetB0 model trained using transfer learning "
        "to classify 6 common oral conditions from a single image."
    )
    st.metric("Test Set Accuracy", "93.4%")
    st.markdown("---")
    st.caption("⚠️ This is an educational project and not a substitute for a professional dental consultation.")

# ============================================================
# Main UI
# ============================================================
st.title("🦷 Oral Disease Classifier")
st.write("Upload a clear image of the mouth or teeth, and the model will classify it into one of 6 possible conditions.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Analyzing..."):
        predictions = model.predict(img_array, verbose=0)[0]

    top_idx = int(np.argmax(predictions))
    top_class = CLASS_NAMES[top_idx]
    top_conf = predictions[top_idx] * 100

    st.markdown(f"""
    <div class="result-box">
        <h3>🔍 Predicted Diagnosis</h3>
        <h2>{top_class}</h2>
        <p><b>Confidence:</b> {top_conf:.1f}%</p>
        <p style="font-size:0.9rem;">{CLASS_INFO[top_class]}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📊 Confidence Across All Classes")
    sorted_idx = np.argsort(predictions)[::-1]
    for idx in sorted_idx:
        cls = CLASS_NAMES[idx]
        conf = predictions[idx] * 100
        st.write(f"**{cls}**")
        st.progress(float(predictions[idx]))
        st.caption(f"{conf:.1f}%")

    st.markdown("---")
    st.caption("⚠️ This result is for educational purposes only and does not replace a consultation with a qualified dentist.")
else:
    st.info("👆 Upload an image to get started")
