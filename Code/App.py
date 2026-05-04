import os
import io
from pathlib import Path

import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Osteosarcoma DCNN App",
    page_icon="🩺",
    layout="wide"
)

# =========================================================
# SETTINGS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
IMG_SIZE = (120, 120)
CONFIDENCE_THRESHOLD = 0.70

TASK_CONFIG = {
    "3-Class Classification": {
        "path": MODEL_DIR / "task_3class_dcnn.keras",
        "labels": ["Non_Tumor", "Non_Viable_Tumor", "Viable_Tumor"],
        "description": "Custom DCNN trained for 3-class osteosarcoma histopathology classification.",
        "input_size": "120 x 120 x 3",
        "preprocessing": "Per-image Z-score normalization",
        "task_type": "multiclass"
    },
    "Tumor vs Non-Tumor": {
        "path": MODEL_DIR / "task_tumor_vs_nontumor_dcnn.keras",
        "labels": ["Non_Tumor", "Tumor"],
        "description": "Custom DCNN trained for binary tumor screening.",
        "input_size": "120 x 120 x 3",
        "preprocessing": "Per-image Z-score normalization",
        "task_type": "binary"
    }
}

# =========================================================
# PREPROCESSING
# =========================================================
def zscore_preprocess(x):
    x = x.astype("float32")
    mean = np.mean(x, axis=(0, 1), keepdims=True)
    std = np.std(x, axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-7, 1.0, std)
    x = (x - mean) / std
    return x

def prepare_image(uploaded_image):
    original = uploaded_image.convert("RGB")
    resized = original.resize(IMG_SIZE)
    img_array = np.array(resized).astype("float32")
    img_array = zscore_preprocess(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return original, resized, img_array

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model(model_path_str):
    model_path = Path(model_path_str)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return tf.keras.models.load_model(model_path)

# =========================================================
# PREDICTION
# =========================================================
def predict_image(model, processed_img, task_type, label_names):
    preds = model.predict(processed_img, verbose=0)

    if task_type == "binary":
        prob_positive = float(preds[0][0])
        prob_negative = 1.0 - prob_positive
        probs = [prob_negative, prob_positive]
    else:
        probs = preds[0].tolist()

    pred_index = int(np.argmax(probs))
    pred_label = label_names[pred_index]
    confidence = float(probs[pred_index])

    return pred_label, probs, confidence, pred_index

# =========================================================
# FIND LAST CONV LAYER
# =========================================================
def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer

    for layer in reversed(model.layers):
        if hasattr(layer, "layers"):
            for sublayer in reversed(layer.layers):
                if isinstance(sublayer, tf.keras.layers.Conv2D):
                    return sublayer

    return None

# =========================================================
# GRAD-CAM
# =========================================================
def make_gradcam_heatmap(img_array, model, pred_index=None):
    _ = model(img_array, training=False)

    last_conv_layer = find_last_conv_layer(model)
    if last_conv_layer is None:
        raise ValueError("No Conv2D layer found for Grad-CAM.")

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)

        if predictions.shape[-1] == 1:
            class_channel = predictions[:, 0]
        else:
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        raise ValueError("Gradients could not be computed.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)

    max_val = tf.reduce_max(heatmap)
    if float(max_val) > 0:
        heatmap /= max_val

    return heatmap.numpy(), last_conv_layer.name

def apply_heatmap_to_image(original_pil_image, heatmap, alpha=0.4):
    original = original_pil_image.convert("RGB").resize(IMG_SIZE)
    original_np = np.array(original).astype("float32") / 255.0

    heatmap_img = Image.fromarray(np.uint8(heatmap * 255)).resize(IMG_SIZE)
    heatmap_np = np.array(heatmap_img).astype("float32") / 255.0

    overlay = np.zeros((IMG_SIZE[1], IMG_SIZE[0], 3), dtype="float32")
    overlay[..., 0] = heatmap_np

    superimposed = np.clip((1 - alpha) * original_np + alpha * overlay, 0, 1)
    return Image.fromarray(np.uint8(superimposed * 255))

# =========================================================
# HELPERS
# =========================================================
def build_result_dataframe(task_name, class_names, probs, pred_label, confidence):
    rows = []
    for cls, prob in zip(class_names, probs):
        rows.append({
            "task": task_name,
            "class": cls,
            "probability": float(prob),
            "predicted_class": pred_label,
            "confidence": float(confidence)
        })
    return pd.DataFrame(rows)

def pil_image_to_bytes(img, fmt="PNG"):
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf

# =========================================================
# HEADER
# =========================================================
st.title("Osteosarcoma Histopathology Classification")
st.caption("DCNN-based research demo for histopathology image classification.")

with st.expander("About this model"):
    st.write("This app uses a custom DCNN model for osteosarcoma histopathology classification.")
    st.write("Supported tasks:")
    st.write("- 3-Class Classification")
    st.write("- Tumor vs Non-Tumor")
    st.write("Input size: 120 x 120 x 3")
    st.write("Preprocessing: Per-image Z-score normalization")

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("Options")

task_name = st.sidebar.selectbox(
    "Choose task",
    list(TASK_CONFIG.keys())
)

show_gradcam = st.sidebar.checkbox("Show Grad-CAM", value=True)
gradcam_alpha = st.sidebar.slider("Grad-CAM overlay strength", 0.1, 0.9, 0.4, 0.1)

uploaded_file = st.file_uploader(
    "Upload histopathology image",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"]
)

# =========================================================
# MAIN
# =========================================================
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        display_img, resized_img, processed_img = prepare_image(image)

        cfg = TASK_CONFIG[task_name]
        model = load_model(str(cfg["path"]))

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Uploaded Image")
            st.image(display_img, caption="Input Image", use_column_width=True)

        with col2:
            st.subheader("Task Info")
            st.write(f"**Task:** {task_name}")
            st.write(f"**Description:** {cfg['description']}")
            st.write(f"**Classes:** {', '.join(cfg['labels'])}")

        if st.button("Predict"):
            pred_label, probs, confidence, pred_index = predict_image(
                model=model,
                processed_img=processed_img,
                task_type=cfg["task_type"],
                label_names=cfg["labels"]
            )

            st.subheader("Prediction Result")

            if confidence < CONFIDENCE_THRESHOLD:
                st.warning(
                    f"Predicted Class: {pred_label} | Confidence: {confidence:.4f}. "
                    f"This is a low-confidence prediction."
                )
            else:
                st.success(f"Predicted Class: {pred_label} | Confidence: {confidence:.4f}")

            st.subheader("Prediction Probabilities")
            prob_df = pd.DataFrame({
                "Class": cfg["labels"],
                "Probability": [float(p) for p in probs]
            }).sort_values("Probability", ascending=False)

            st.dataframe(prob_df, use_container_width=True)
            st.bar_chart(prob_df.set_index("Class"))

            if len(prob_df) > 1:
                st.subheader("Top Predictions")
                st.write(f"**Top-1:** {prob_df.iloc[0]['Class']} ({prob_df.iloc[0]['Probability']:.4f})")
                st.write(f"**Top-2:** {prob_df.iloc[1]['Class']} ({prob_df.iloc[1]['Probability']:.4f})")

            if show_gradcam:
                st.subheader("Grad-CAM Visualization")
                try:
                    gradcam_pred_index = pred_index if cfg["task_type"] != "binary" else None
                    heatmap, last_conv_name = make_gradcam_heatmap(
                        processed_img,
                        model,
                        pred_index=gradcam_pred_index
                    )
                    gradcam_img = apply_heatmap_to_image(display_img, heatmap, alpha=gradcam_alpha)

                    g1, g2 = st.columns(2)
                    with g1:
                        st.image(display_img.resize(IMG_SIZE), caption="Original", use_column_width=True)
                    with g2:
                        st.image(gradcam_img, caption=f"Grad-CAM ({last_conv_name})", use_column_width=True)

                    gradcam_buf = pil_image_to_bytes(gradcam_img, fmt="PNG")
                    st.download_button(
                        label="Download Grad-CAM",
                        data=gradcam_buf,
                        file_name=f"gradcam_{task_name.replace(' ', '_').lower()}.png",
                        mime="image/png"
                    )
                except Exception as gradcam_error:
                    st.info(f"Grad-CAM unavailable: {gradcam_error}")

            st.subheader("Download Results")
            result_df = build_result_dataframe(
                task_name=task_name,
                class_names=cfg["labels"],
                probs=probs,
                pred_label=pred_label,
                confidence=confidence
            )

            csv_bytes = result_df.to_csv(index=False).encode("utf-8")
            json_bytes = result_df.to_json(orient="records", indent=2).encode("utf-8")

            d1, d2 = st.columns(2)
            with d1:
                st.download_button(
                    label="Download CSV",
                    data=csv_bytes,
                    file_name=f"prediction_{task_name.replace(' ', '_').lower()}.csv",
                    mime="text/csv"
                )
            with d2:
                st.download_button(
                    label="Download JSON",
                    data=json_bytes,
                    file_name=f"prediction_{task_name.replace(' ', '_').lower()}.json",
                    mime="application/json"
                )

    except Exception as e:
        st.error(f"Error: {str(e)}")

st.markdown("---")
st.caption("Research-use demo only. Not for clinical diagnosis.")