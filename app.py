import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)

# ---------------- UI CONFIG ----------------
st.set_page_config(
    page_title="AI Vision Classifier",
    page_icon="🧠",
    layout="centered"
)

# ---------------- HEADER ----------------
st.markdown(
    "<h1 style='text-align:center;'>🧠 AI Vision Classifier</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;color:gray;'>Powered by MobileNetV2 + TensorFlow</p>",
    unsafe_allow_html=True
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return MobileNetV2(weights="imagenet")

model = load_model()

# ---------------- LABEL CLEANING ----------------
def clean_label(label):
    label = label.lower()

    # Horse group
    if "horse" in label or label == "sorrel":
        return "Horse 🐎"

    # Cat group
    if "cat" in label:
        return "Cat 🐱"

    # Dog group
    if "dog" in label:
        return "Dog 🐶"

    # Bird group
    if "bird" in label:
        return "Bird 🐦"

    # Default cleanup
    return label.replace("_", " ").title()

# ---------------- UPLOAD ----------------
uploaded = st.file_uploader(
    "📤 Upload an image",
    type=["jpg", "jpeg", "png"]
)

# ---------------- PROCESS ----------------
if uploaded:

    col1, col2 = st.columns(2)

    image = Image.open(uploaded)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img = image.resize((224,224))
    img = np.array(img)

    if img.shape[-1] == 4:
        img = img[:,:,:3]

    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    # Prediction
    pred = model.predict(img)
    results = decode_predictions(pred, top=3)[0]

    # ---------------- OUTPUT ----------------
    with col2:
        st.markdown("### 🔍 Predictions")

        for i, (_, label, score) in enumerate(results):

            confidence = float(score * 100)
            clean = clean_label(label)

            st.markdown(
                f"""
                **{i+1}. {clean}**  
                Confidence: {confidence:.2f}%
                """
            )

            st.progress(int(confidence))

    # ---------------- FINAL RESULT ----------------
    best_label = clean_label(results[0][1])
    best_score = results[0][2] * 100

    st.success(f"🎯 Final Prediction: {best_label} ({best_score:.2f}%)")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:gray;'>Made with ❤️ using Streamlit + MobileNetV2</p>",
    unsafe_allow_html=True
)