# app.py
import joblib
import numpy as np
import streamlit as st

# ----------------------------
# Config
# ----------------------------
MODEL_PATH = "spam_pipeline.joblib"
PAGE_TITLE = "📧 Spam vs Ham — Classifier"

# ----------------------------
# Helpers
# ----------------------------
@st.cache_resource(show_spinner=True)
def load_model(path: str):
    try:
        model = joblib.load(path)
        _ = hasattr(model, "predict")
        return model, None
    except Exception as e:
        return None, e


def normalize_label(raw_label):
    if isinstance(raw_label, (np.generic, np.bool_)):
        raw_label = raw_label.item()
    s = str(raw_label).strip().lower()
    if s in {"spam", "1", "true", "yes"} or raw_label == 1 or raw_label is True:
        return "SPAM", True
    return "HAM", False


def predict_label(model, text: str):
    try:
        pred = model.predict([text])[0]
        proba = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba([text])[0]
            proba = float(np.max(probs))
        verdict, is_spam = normalize_label(pred)
        return verdict, proba, None
    except Exception as e:
        return None, None, e


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title=PAGE_TITLE, page_icon="📧", layout="centered")
st.title(PAGE_TITLE)

model, load_err = load_model(MODEL_PATH)
if load_err:
    st.error("Failed to load model.")
    st.exception(load_err)
    st.stop()

# Example messages
default_ham = "Hey, are we still on for lunch at 1 pm? The cafe near the office."
default_spam = "CONGRATULATIONS! You won ₹10,00,000. Click here to claim now!!!"

col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    if st.button("Use Ham example"):
        st.session_state["msg"] = default_ham
with col2:
    if st.button("Use Spam example"):
        st.session_state["msg"] = default_spam
with col3:
    st.caption("Tip: use examples to sanity-check the model quickly.")

text = st.text_area(
    "Enter message to classify",
    value=st.session_state.get("msg", ""),
    height=160,
    placeholder="Paste or type the email/SMS text here…",
)

# Classification
if st.button("Classify", type="primary"):
    if not text.strip():
        st.warning("Please enter some text.")
        st.stop()

    label, prob, pred_err = predict_label(model, text)
    if pred_err:
        st.error("Prediction failed.")
        st.exception(pred_err)
        st.stop()

    if label == "SPAM":
        st.error(f"Prediction: **{label}**")
    else:
        st.success(f"Prediction: **{label}**")

    if prob is not None:
        st.caption(f"Confidence: {prob:.2%}")

    with st.expander("Details"):
        st.write("Raw text:")
        st.code(text)
        if hasattr(model, "classes_") and hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba([text])[0]
                table = {str(c): float(p) for c, p in zip(model.classes_, probs)}
                st.json(table)
            except Exception as e:
                st.info("Could not compute per-class probabilities:")
                st.exception(e)

st.markdown("---")
st.caption("Built with Streamlit • Simple Spam/Ham classifier")
