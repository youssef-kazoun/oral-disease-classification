import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import matplotlib.cm as cm

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
# Grad-CAM helpers
# ============================================================
@st.cache_resource
def get_gradcam_layers(_model):
    """Extract the sub-layers needed to manually run the forward pass for Grad-CAM."""
    rescaling_layer = _model.get_layer('rescaling_8')
    base_model_layer = _model.get_layer('efficientnetb0')
    gap_layer = _model.get_layer('global_average_pooling2d_5')
    dense1_layer = _model.get_layer('dense_10')
    dropout_layer = _model.get_layer('dropout_5')
    dense2_layer = _model.get_layer('dense_11')
    return rescaling_layer, base_model_layer, gap_layer, dense1_layer, dropout_layer, dense2_layer

def make_gradcam_heatmap(img_array, layers, pred_index=None):
    rescaling_layer, base_model_layer, gap_layer, dense1_layer, dropout_layer, dense2_layer = layers

    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    rescaled = rescaling_layer(img_tensor)

    with tf.GradientTape() as tape:
        conv_outputs = base_model_layer(rescaled, training=False)
        tape.watch(conv_outputs)
        x = gap_layer(conv_outputs)
        x = dense1_layer(x)
        x = dropout_layer(x, training=False)
        predictions = dense2_layer(x)

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_gradcam(original_img, heatmap, alpha=0.4):
    img = np.array(original_img.resize((224, 224)))

    heatmap_uint8 = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_uint8]

    jet_heatmap = Image.fromarray(np.uint8(jet_heatmap * 255))
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = np.array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    return Image.fromarray(superimposed_img)

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
    image = Image.open(uploaded_file).convert("RGB")

    # Preprocess
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("جارٍ التحليل..."):
        predictions = model.predict(img_array, verbose=0)[0]
        top_idx = int(np.argmax(predictions))

        # Grad-CAM
        try:
            layers = get_gradcam_layers(model)
            heatmap = make_gradcam_heatmap(img_array, layers, pred_index=top_idx)
            gradcam_img = overlay_gradcam(image, heatmap)
            gradcam_available = True
        except Exception:
            gradcam_available = False

    top_class = CLASS_NAMES[top_idx]
    top_conf = predictions[top_idx] * 100

    # ---- Images side by side ----
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="الصورة المرفوعة", use_container_width=True)
    with col2:
        if gradcam_available:
            st.image(gradcam_img, caption="🔥 Grad-CAM: أين ينظر الموديل؟", use_container_width=True)
        else:
            st.info("تعذّر توليد Grad-CAM لهذه الصورة")

    # ---- Result box ----
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
