# Run in terminal:
# streamlit run app.py

import streamlit as st
import numpy as np
import pickle
from PIL import Image
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Handwritten Digit Recognition AI",
    page_icon="",
    layout="centered"
)

@st.cache_resource
def load_all_models():
    cnn_model = load_model("cnn_model.h5")
    nn_model = load_model("neural_network_model.h5")

    with open("logistic_model.pkl", "rb") as f:
        log_model = pickle.load(f)

    with open("knn_model.pkl", "rb") as f:
        knn_model = pickle.load(f)

    return cnn_model, nn_model, log_model, knn_model

cnn_model, nn_model, log_model, knn_model = load_all_models()

st.sidebar.title("Settings")
model_choice = st.sidebar.selectbox(
    "Select AI Model",
    (
        "Logistic Regression",
        "KNN",
        "Neural Network",
        "CNN (Best Accuracy)"
    )
)

st.sidebar.markdown("---")
st.sidebar.write("(1) Upload a clear handwritten digit image")
st.sidebar.write("(2) Background: Black | Digit: White")
st.sidebar.write("(3) Size does not matter (auto resize)")
st.sidebar.write("(4) Upload only a single handwritten digit image (0–9)")

st.markdown(
    "<h1 style='text-align: center;'>Handwritten Digit Recognition AI</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center;'>Upload your handwritten digit (0–9) and let AI predict it</p>",
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    image = image.resize((28, 28))

    img = np.array(image)
    img = img / 255.0

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", width=200)

    with col2:
        st.markdown("Prediction Result")

        if model_choice == "Logistic Regression":
            img_flat = img.reshape(1, 784)
            digit = log_model.predict(img_flat)[0]

        elif model_choice == "KNN":
            img_flat = img.reshape(1, 784)
            digit = knn_model.predict(img_flat)[0]

        elif model_choice == "Neural Network":
            img_nn = img.reshape(1, 28, 28)
            pred = nn_model.predict(img_nn)
            digit = np.argmax(pred)

        elif model_choice == "CNN (Best Accuracy)":
            img_cnn = img.reshape(1, 28, 28, 1)
            pred = cnn_model.predict(img_cnn)
            digit = np.argmax(pred)

        st.success(f"Model Used: {model_choice}")
        st.success(f"Predicted Digit: {digit}")

st.markdown("---")
st.markdown(
    "<p style='text-align: center;'>Built with Python, ML & Deep Learning</p>",
    unsafe_allow_html=True
)

