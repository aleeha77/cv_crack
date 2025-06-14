import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

model = load_model("crack_model_subset.h5")

st.title("Concrete Crack Detection")
st.write("Upload a concrete image. The model will predict if there's a crack.")

uploaded_file = st.file_uploader("Upload a concrete image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img = img.resize((100, 100))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    label = "🧱 Crack Detected" if prediction > 0.5 else "✅ No Crack Detected"

    st.markdown(f"### {label}")
    st.markdown(f"**Confidence:** `{prediction:.2f}`")
