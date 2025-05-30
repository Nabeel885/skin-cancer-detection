import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
import shutil

# Page setup
st.set_page_config(page_title="Cancer Detection", page_icon="🧬", layout="centered")

# Load model
model = YOLO("best.pt")

# Custom CSS for modern UI
st.markdown("""
<style>
body {
    font-family: 'Segoe UI', sans-serif;
    background-color: #f2f6fc;
}
.main {
    background-color: #ffffff;
    padding: 2rem;
    border-radius: 15px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.08);
    margin-top: 30px;
}
h1 {
    color: #0f4c75;
    font-weight: 700;
    text-align: center;
}
.subheading {
    color: #555555;
    text-align: center;
    margin-bottom: 1.5rem;
}
.label-green {
    font-weight: 700;
    font-size: 1.1em;
    color: white;
    background-color: #28a745;
    padding: 4px 8px;
    border-radius: 6px;
    display: inline-block;
    margin-bottom: 0.3em;
}
.heading-left {
    text-align: left;
    color: #0f4c75;
    font-weight: 700;
    font-size: 1.4em;
    margin-top: 1rem;
    margin-bottom: 0.5em;
}
.stButton > button {
    background-color: #0f4c75;
    color: white;
    border-radius: 8px;
    padding: 0.6rem 1.2rem;
    border: none;
    font-weight: 600;
    transition: all 0.3s ease;
}
.stButton > button:hover {
    background-color: #3282b8;
    transform: scale(1.03);
}
.footer {
    text-align: center;
    margin-top: 3rem;
    font-size: 13px;
    color: #888888;
}
</style>
""", unsafe_allow_html=True)

# UI container
st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown('<h1>🧪 Cancer Detection using YOLOv8</h1>', unsafe_allow_html=True)
st.markdown('<p class="subheading">Upload a skin image to detect potential cancerous areas</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📁 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    # Clear previous results
    if os.path.exists("outputs"):
        shutil.rmtree("outputs")

    with st.spinner("🧠 Analyzing with YOLOv8..."):
        # Run prediction
        results = model(img_array)

        # Save prediction images
        os.makedirs("outputs", exist_ok=True)
        save_paths = []
        for i, r in enumerate(results):
            annotated_img = r.plot()
            save_path = os.path.join("outputs", f"result_{i}.jpg")
            Image.fromarray(annotated_img).save(save_path)
            save_paths.append(save_path)

    if save_paths:
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="📷 Original Image", use_column_width=True)
            st.markdown('<div class="heading-left">📌 Predictions</div>', unsafe_allow_html=True)
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                st.markdown(f'<div class="label-green">{label}</div>', unsafe_allow_html=True)

        with col2:
            st.image(save_paths[0], caption="✅ Prediction Result", use_column_width=True)

    else:
        st.warning("Prediction image not found.")

st.markdown('<div class="footer">Made with 🤍 using Streamlit and YOLOv8</div>', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
