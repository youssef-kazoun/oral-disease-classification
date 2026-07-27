
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
    "Calculus": "ترسبات كلسية صلبة على الأسنان ناتجة عن تراكم البلاك",
    "Caries": "تسوس الأسنان الناتج عن تآكل طبقة المينا",
    "Gingivitis": "التهاب اللثة، غالبًا بسبب تراكم البلاك البكتيري",
    "Hypodontia": "غياب خلقي لواحد أو أكثر من الأسنان",
    "Mouth Ulcer": "قرحة أو تقرح في الأنسجة الرخوة للفم",
    "Tooth Discoloration": "تغير لون الأسنان عن اللون الطبيعي",
}

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.header("ℹ️ عن المشروع")
    st.write(
        "موديل EfficientNetB0 مدرب باستخدام Transfer Learning "
        "لتصنيف 6 حالات فموية شائعة من صورة واحدة."
    )
    st.metric("دقة الموديل على بيانات الاختبار", "93.4%")
    st.markdown("---")
    st.caption("⚠️ هذا مشروع تعليمي وليس بديلاً عن استشارة طبيب الأسنان.")

# ============================================================
# Main UI
# ============================================================
st.title("🦷 Oral Disease Classifier")
st.write("ارفع صورة واضحة للفم أو الأسنان، وسيقوم الموديل بتصنيفها إلى إحدى 6 حالات محتملة.")

uploaded_file = st.file_uploader("اختر صورة...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])

    image = Image.open(uploaded_file).convert("RGB")

    with col1:
        st.image(image, caption="الصورة المرفوعة", use_container_width=True)

    # Preprocess
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("جارٍ التحليل..."):
        predictions = model.predict(img_array, verbose=0)[0]

    top_idx = np.argmax(predictions)
    top_class = CLASS_NAMES[top_idx]
    top_conf = predictions[top_idx] * 100

    with col2:
        st.markdown(f"""
        <div class="result-box">
            <h3>🔍 التشخيص المتوقع</h3>
            <h2 style="color:#2E86AB;">{top_class}</h2>
            <p><b>نسبة الثقة:</b> {top_conf:.1f}%</p>
            <p style="font-size:0.9rem; color:#555;">{CLASS_INFO[top_class]}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📊 توزيع الثقة على كل الحالات")
    sorted_idx = np.argsort(predictions)[::-1]
    for idx in sorted_idx:
        cls = CLASS_NAMES[idx]
        conf = predictions[idx] * 100
        st.write(f"**{cls}**")
        st.progress(float(predictions[idx]))
        st.caption(f"{conf:.1f}%")

    st.markdown("---")
    st.caption("⚠️ هذه النتيجة تعليمية فقط ولا تُغني عن استشارة طبيب أسنان مختص.")
else:
    st.info("👆 قم برفع صورة للبدء")
