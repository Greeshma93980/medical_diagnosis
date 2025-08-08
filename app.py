import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

st.title("🩺 Pneumonia Detection from Chest X-rays")
st.markdown("Upload a chest X-ray image to detect **Pneumonia** using a deep learning model.")

uploaded_file = st.file_uploader("📤 Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    # Preprocess image
    image = image.resize((150, 150))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Load and run model
    model = tf.keras.models.load_model("pneumonia_model.h5")
    _ = model(tf.zeros((1, 150, 150, 3)))  # Force model to build
    prediction = model.predict(img_array)[0][0]

    # Display result
    if prediction > 0.5:
        st.error(f"🔴 **Prediction: Pneumonia** ({prediction * 100:.2f}% confidence)")
    else:
        st.success(f"🟢 **Prediction: Normal** ({(1 - prediction) * 100:.2f}% confidence)")
