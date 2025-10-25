# app.py
import streamlit as st
import joblib
import numpy as np
from pathlib import Path
import requests
from typing import Optional

# ---------- Page config ----------
st.set_page_config(
    page_title="Language Detector",
    layout="centered",
    initial_sidebar_state="auto",
)

# ---------- Config ----------
MODEL_PATH = "lang_detector_pipeline.joblib"
DOWNLOAD_MODEL_IF_MISSING = True
REMOTE_MODEL_URL: str = ""  # set a direct-download URL if you host the model remotely

LANG_MAP = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "zh": "Chinese",
}

# ---------- Utilities ----------
@st.cache_resource(show_spinner=False)
def download_model(url: str, dest: Path) -> bool:
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        dest.write_bytes(r.content)
        return True
    except Exception as e:
        st.error(f"Failed to download model: {e}")
        return False

@st.cache_resource(show_spinner=False)
def load_model(path: str):
    p = Path(path)
    if not p.exists():
        if DOWNLOAD_MODEL_IF_MISSING and REMOTE_MODEL_URL:
            ok = download_model(REMOTE_MODEL_URL, p)
            if not ok:
                return None
        else:
            st.error(f"Model file not found at: {path}")
            return None
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

def pretty_confidence(prob: float) -> str:
    return f"{prob*100:.1f}%"

def predict_single(model, text: str) -> Optional[tuple]:
    """Return (pred_code, confidence) or None on failure."""
    try:
        if not hasattr(model, "predict_proba"):
            pred = model.predict([text])[0]
            return pred, None
        proba = model.predict_proba([text])[0]
        pred = model.predict([text])[0]
        top_idx = int(np.argmax(proba))
        confidence = float(proba[top_idx])
        return pred, confidence
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# ---------- Sidebar (info & settings) ----------
with st.sidebar:
    st.header("About")
    st.write(
        "Language Detector\n\n"
        "Detects the language of input text using a TF-IDF (char n-grams) + Logistic Regression pipeline."
    )
    st.divider()
    st.subheader("Model")
    st.write(f"Expected model file: `{MODEL_PATH}`")
    if REMOTE_MODEL_URL:
        st.write("Model will be downloaded automatically if missing.")
    st.divider()
    # st.caption("For production deployments, host the model externally (S3/GCS) and set REMOTE_MODEL_URL.")

# ---------- Main app ----------
st.markdown(
    """
    <style>
    .card {
        background-color: #0f1724;
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 6px 18px rgba(6,9,20,0.6);
        color: #e6eef8;
    }
    .muted { color: #9aa7b2; font-size: 0.95rem; }
    .btn-primary {
        background-color: #ff6b6b;
        border: none;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Language Detector")
st.write("Paste or type text below and press **Detect**. The model will return the single most likely language and a confidence score.")

# use session_state for persistent input
if "text_input" not in st.session_state:
    st.session_state["text_input"] = ""

# helper callback setters for examples and clear
def set_example_en():
    st.session_state["text_input"] = "This is a simple English sentence used as an example."

def set_example_es():
    st.session_state["text_input"] = "Hola, ¿cómo estás? Este es un ejemplo."

def set_example_fr():
    st.session_state["text_input"] = "Ceci est un exemple de phrase en français."

def set_example_hi():
    st.session_state["text_input"] = "यह एक उदाहरण वाक्य है।"

def set_example_zh():
    st.session_state["text_input"] = "这是一个中文示例句子。"

def clear_input():
    st.session_state["text_input"] = ""

col_left, col_right = st.columns([3, 1])

with col_left:
    input_val = st.text_area(
        label="Enter text to identify language",
        value=st.session_state["text_input"],
        height=220,
        placeholder="Type or paste text here..."
    )
    # keep session state in sync
    st.session_state["text_input"] = input_val

    st.write("")  # spacer
    btn_col1, btn_col2 = st.columns([1, 1])
    with btn_col1:
        detect_pressed = st.button("Detect", type="primary")
    with btn_col2:
        clear_pressed = st.button("Clear", on_click=clear_input, type="secondary")

with col_right:
    st.subheader("Examples")
    st.button("English example", on_click=set_example_en)
    st.button("Spanish example", on_click=set_example_es)
    st.button("French example", on_click=set_example_fr)
    st.button("Hindi example", on_click=set_example_hi)
    st.button("Chinese example", on_click=set_example_zh)

# Load model (cached)
with st.spinner("Loading model..."):
    model = load_model(MODEL_PATH)

if model is None:
    st.stop()

# If Detect clicked, run prediction and show a polished result card
if detect_pressed:
    text_to_check = st.session_state.get("text_input", "").strip()
    if not text_to_check:
        st.warning("Please enter some text to detect.")
    else:
        result = predict_single(model, text_to_check)
        if result is None:
            st.error("Prediction failed.")
        else:
            pred_code, confidence = result
            lang_name = LANG_MAP.get(pred_code, pred_code)

            # Professional result card
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Detected language")
            st.markdown(f"### {lang_name}  •  `{pred_code}`")
            if confidence is None:
                st.info("Confidence: N/A for this model")
            else:
                st.metric(label="Confidence", value=pretty_confidence(confidence))
                pct = int(confidence * 100)
                st.progress(pct)
            st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("Language Detector")
