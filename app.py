import json
import os

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Set MODEL_PATH env var to point at wherever you actually saved the .h5 file
# (e.g. `docker run -e MODEL_PATH=/models/plant_disease_model.h5 ...`).
# Defaults to models/plant_disease_model.h5 inside this project.
MODEL_PATH = os.environ.get(
    "MODEL_PATH", os.path.join("models", "plant_disease_model.h5")
)
CLASS_INDICES_PATH = "class_indices.json"
IMAGE_SIZE = (256, 256)

# The model file is never committed to GitHub (too big for a normal push).
# Instead, keep it on Google Drive and let the app pull it down the first
# time it starts. Set this to your Drive file's ID (see README section 0).
# Works from either an environment variable or Streamlit secrets, so it
# works the same locally, in Docker, and on Streamlit Community Cloud.
GDRIVE_FILE_ID = os.environ.get("GDRIVE_FILE_ID") or st.secrets.get(
    "GDRIVE_FILE_ID", ""
)

st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="centered",
)


# ---------------------------------------------------------------------------
# Download the model from Google Drive on first run, if it isn't already
# sitting on disk. This is what lets you push just the code to GitHub
# (small, fast) while the ~large .h5 file stays on Drive and gets fetched
# automatically whenever the app boots.
# ---------------------------------------------------------------------------
def ensure_model_downloaded() -> bool:
    if os.path.exists(MODEL_PATH):
        return True
    if not GDRIVE_FILE_ID:
        return False

    import gdown

    os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)
    with st.spinner("Downloading model from Google Drive (first run only)..."):
        gdown.download(
            id=GDRIVE_FILE_ID, output=MODEL_PATH, quiet=False
        )
    return os.path.exists(MODEL_PATH)


# ---------------------------------------------------------------------------
# Cached loaders (only run once per app session)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    if not ensure_model_downloaded():
        return None
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_class_indices():
    with open(CLASS_INDICES_PATH, "r") as f:
        raw = json.load(f)
    # JSON keys are always strings -> convert back to int keys
    return {int(k): v for k, v in raw.items()}


def preprocess_image(img: Image.Image, target_size=IMAGE_SIZE) -> np.ndarray:
    img = img.convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict(model, class_indices, img: Image.Image):
    preprocessed = preprocess_image(img)
    predictions = model.predict(preprocessed)
    predicted_index = int(np.argmax(predictions, axis=1)[0])
    confidence = float(np.max(predictions))
    predicted_label = class_indices.get(predicted_index, f"Class {predicted_index}")
    return predicted_label, confidence, predictions[0]


def pretty_label(label: str) -> str:
    return label.replace("___", " — ").replace("_", " ")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🌿 Plant Disease Classifier")
st.write(
    "Upload a photo of a plant leaf and the model will predict whether it's "
    "healthy or which disease it may have."
)

model = load_model()

if model is None:
    if GDRIVE_FILE_ID:
        st.error(
            "Couldn't download or load the model from Google Drive "
            f"(file id `{GDRIVE_FILE_ID}`). Make sure the Drive file is "
            "shared as 'Anyone with the link' and that GDRIVE_FILE_ID is "
            "correct, then restart the app."
        )
    else:
        st.error(
            f"Model file not found at `{MODEL_PATH}`, and no `GDRIVE_FILE_ID` "
            "is set.\n\n"
            "Easiest fix: set the `GDRIVE_FILE_ID` secret/env var so the app "
            "downloads it from Google Drive automatically (see README "
            "section 0). Or, do it manually:\n"
            "- place `plant_disease_model.h5` inside the `models/` folder of "
            "this project, or\n"
            "- set the `MODEL_PATH` environment variable to point at wherever "
            "you saved it (e.g. `docker run -e MODEL_PATH=/path/to/model.h5 ...`), "
            "then restart the app."
        )
    st.stop()

try:
    class_indices = load_class_indices()
except FileNotFoundError:
    st.error(
        f"`{CLASS_INDICES_PATH}` not found. This file maps model output indices "
        "to class names and is required for predictions to make sense. "
        "See the README for how to regenerate it from your notebook."
    )
    st.stop()

uploaded_file = st.file_uploader(
    "Choose a leaf image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image")

    if st.button("Predict"):
        with st.spinner("Analyzing leaf..."):
            label, confidence, all_probs = predict(model, class_indices, image)

        st.subheader("Result")
        st.success(f"**{pretty_label(label)}**")
        st.write(f"Confidence: **{confidence * 100:.2f}%**")

        # Show top 5 predictions
        top5_idx = np.argsort(all_probs)[::-1][:5]
        st.write("Top 5 predictions:")
        for idx in top5_idx:
            name = class_indices.get(int(idx), f"Class {idx}")
            st.write(f"- {pretty_label(name)}: {all_probs[idx] * 100:.2f}%")

st.markdown("---")
st.caption(
    "Model trained on the PlantVillage dataset using a CNN "
    "(Conv2D + MaxPooling + Dense layers, Keras/TensorFlow)."
)
