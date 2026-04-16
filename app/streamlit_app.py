import streamlit as st

# MUST be first Streamlit command
st.set_page_config(page_title="AI Real Estate Agent", layout="wide")

import sys
import traceback
from pathlib import Path

# Add the parent directory to sys.path so we can import 'app' modules
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Now import from the 'app' package
BACKEND_AVAILABLE = False
model_pipeline = None
train_stats = None
import_error_msg = None

try:
    from app.llm_chain import extract_features as extract_features_direct, generate_interpretation
    from app.utils import load_model, load_stats, prepare_features_for_prediction
    import numpy as np
    BACKEND_AVAILABLE = True
except ImportError as e:
    import_error_msg = f"Import error: {e}\n{traceback.format_exc()}"
except Exception as e:
    import_error_msg = f"Unexpected error during import: {e}"

# Try to load model (only if imports succeeded)
if BACKEND_AVAILABLE:
    try:
        model_pipeline = load_model()
        train_stats = load_stats()
    except Exception as e:
        BACKEND_AVAILABLE = False
        import_error_msg = f"Model loading failed: {e}"

# ============================================================================
# UI
# ============================================================================
st.title("🏡 AI Real Estate Agent")
st.markdown("Describe a property, review features, get AI‑powered price prediction.")

if import_error_msg:
    with st.expander("🔧 Debug Info (click to expand)"):
        st.error(import_error_msg)

# API fallback URL (only used if BACKEND_AVAILABLE is False)
API_URL = "http://localhost:8000"

# Session state
if "extraction" not in st.session_state:
    st.session_state.extraction = None
if "edited_features" not in st.session_state:
    st.session_state.edited_features = {}

# Input
query = st.text_input(
    "Your property description:",
    placeholder="e.g., 3 bed, 2 bath ranch with 2‑car garage, 1800 sqft"
)
prompt_version = st.selectbox("Prompt version:", ["v1", "v2"], index=0)

# ============================================================================
# Extraction
# ============================================================================
if st.button("Extract Features") and query:
    with st.spinner("Analyzing..."):
        if BACKEND_AVAILABLE:
            try:
                extraction_result = extract_features_direct(query, prompt_version)
                st.session_state.extraction = extraction_result.dict()
                st.session_state.edited_features = {
                    feat: (data["value"] if data["value"] is not None else "")
                    for feat, data in st.session_state.extraction["features"].items()
                }
            except Exception as e:
                st.error(f"Extraction failed: {e}")
        else:
            import requests
            try:
                resp = requests.post(
                    f"{API_URL}/extract",
                    json={"query": query, "prompt_version": prompt_version},
                    timeout=10
                )
                if resp.status_code == 200:
                    st.session_state.extraction = resp.json()
                    st.session_state.edited_features = {
                        feat: (data["value"] if data["value"] is not None else "")
                        for feat, data in st.session_state.extraction["features"].items()
                    }
                else:
                    st.error(f"HTTP extraction failed: {resp.text}")
            except Exception as e:
                st.error(f"HTTP request failed: {e}")

# ============================================================================
# Editing Form
# ============================================================================
if st.session_state.extraction:
    st.subheader("Review & Edit Features")
    missing = st.session_state.extraction.get("missing_features", [])
    if missing:
        st.warning(f"Missing: {', '.join(missing)}. Please fill them.")

    cols = st.columns(2)
    updated_features = {}
    for i, (feat, data) in enumerate(st.session_state.extraction["features"].items()):
        with cols[i % 2]:
            conf = data["confidence"]
            help_text = data.get("reasoning", "")
            current = st.session_state.edited_features.get(feat, "")
            if feat == "HouseStyle":
                options = ["1Story", "2Story", "1.5Fin", "1.5Unf", "SFoyer", "SLvl", "2.5Fin", "2.5Unf"]
                idx = options.index(current) if current in options else 0
                val = st.selectbox(f"{feat} ({conf})", options, index=idx, help=help_text)
            else:
                val = st.text_input(f"{feat} ({conf})", value=str(current), help=help_text)
            updated_features[feat] = val
    st.session_state.edited_features = updated_features

    # ========================================================================
    # Prediction
    # ========================================================================
    if st.button("Predict Price", type="primary"):
        payload = {k: (v if v != "" else None) for k, v in st.session_state.edited_features.items()}
        with st.spinner("Predicting..."):
            if BACKEND_AVAILABLE:
                try:
                    input_df = prepare_features_for_prediction(payload)
                    pred = model_pipeline.predict(input_df)[0]
                    pred_price = float(pred)
                    interp = generate_interpretation(payload, pred_price, train_stats)
                    st.success(f"### Predicted Price: ${pred_price:,.0f}")
                    st.info(interp)
                    with st.expander("Market Stats"):
                        st.metric("Median", f"${train_stats['median']:,.0f}")
                        st.metric("Typical Range", f"${train_stats['q1']:,.0f} – ${train_stats['q3']:,.0f}")
                except Exception as e:
                    st.error(f"Prediction error: {e}")
            else:
                import requests
                try:
                    resp = requests.post(
                        f"{API_URL}/predict",
                        json={"features": payload, "query": query},
                        timeout=10
                    )
                    if resp.status_code == 200:
                        pred = resp.json()
                        st.success(f"### Predicted Price: ${pred['predicted_price']:,.0f}")
                        st.info(pred["interpretation"])
                    else:
                        st.error(f"HTTP prediction failed: {resp.text}")
                except Exception as e:
                    st.error(f"HTTP request failed: {e}")