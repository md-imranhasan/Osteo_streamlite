import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Osteosarcoma Detection App",
    page_icon="🩺",
    layout="centered"
)

# =========================================================
# MODEL PATHS
# CHANGE THESE TO YOUR REAL MODEL PATHS
# =========================================================
MODEL_PATHS = {
    "3-Class Classification": r"E:\Purdue Spring 2026\Spring 2026 BIOL 59500\comparison_outputs\task_3class\DCNN\DCNN_final.keras",
    "Tumor vs Non-Tumor": r"E:\Purdue Spring 2026\Spring 2026 BIOL 59500\comparison_outputs\task_tumor_vs_nontumor\DCNN\DCNN_final.keras"
}

# =========================================================
# LABEL MAPS
# =========================================================
LABEL_MAPS = {
    "3-Class Classification": ["Non_Tumor", "Non_Viable_Tumor", "Viable_Tumor"],
    "Tumor vs Non-Tumor": ["Non_Tumor", "Tumor"]
}

IMG_SIZE = (120, 120)

# =========================================================
# PREPROCESSING
# SAME STYLE USED IN YOUR DCNN CODE
# =========================================================
def zscore_preprocess(x):
    x = x.astype("float32")
    mean = np.mean(x, axis=(0, 1), keepdims=True)
    std = np.std(x, axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-7, 1.0, std)
    x = (x - mean) / std
    return x

def prepare_image(uploaded_image):
    image = uploaded_image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image).astype("float32")
    img_array = zscore_preprocess(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return image, img_array

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

# =========================================================
# PREDICTION
# =========================================================
def predict_image(model, processed_img, task_name):
    preds = model.predict(processed_img, verbose=0)

    if task_name == "Tumor vs Non-Tumor":
        prob_tumor = float(preds[0][0])
        prob_non_tumor = 1.0 - prob_tumor

        probs = [prob_non_tumor, prob_tumor]
        pred_index = int(np.argmax(probs))
        pred_label = LABEL_MAPS[task_name][pred_index]

    else:
        probs = preds[0].tolist()
        pred_index = int(np.argmax(probs))
        pred_label = LABEL_MAPS[task_name][pred_index]

    return pred_label, probs

# =========================================================
# UI
# =========================================================
st.title("Osteosarcoma Histopathology Classification")
st.write("Upload an image and choose a trained DCNN task.")

task_name = st.selectbox(
    "Choose task",
    ["3-Class Classification", "Tumor vs Non-Tumor"]
)

uploaded_file = st.file_uploader(
    "Upload histopathology image",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        display_img, processed_img = prepare_image(image)

        st.subheader("Uploaded Image")
        st.image(display_img, caption="Input Image", use_container_width=True)

        model_path = MODEL_PATHS[task_name]
        model = load_model(model_path)

        if st.button("Predict"):
            pred_label, probs = predict_image(model, processed_img, task_name)

            st.subheader("Prediction Result")
            st.success(f"Predicted Class: {pred_label}")

            st.subheader("Prediction Probabilities")

            class_names = LABEL_MAPS[task_name]
            for class_name, prob in zip(class_names, probs):
                st.write(f"**{class_name}:** {prob:.4f}")
                st.progress(float(prob))

    except Exception as e:
        st.error(f"Error: {str(e)}")