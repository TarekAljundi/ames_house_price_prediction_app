import streamlit as st
import sys
from pathlib import Path

# ============================================================================
# 1. MUST BE THE VERY FIRST STREAMLIT COMMAND
# ============================================================================
st.set_page_config(page_title="AI Real Estate Agent", layout="wide")

# ============================================================================
# 2. Now we can safely import everything else
# ============================================================================
import requests
import json
import numpy as np

sys.path.append(str(Path(__file__).parent))

BACKEND_AVAILABLE = False
model_pipeline = None
train_stats = None

try:
    from llm_chain import extract_features as extract_features_direct, generate_interpretation
    from utils import load_model, load_stats, prepare_features_for_prediction
    BACKEND_AVAILABLE = True
    # Load model eagerly only if backend is available
    model_pipeline = load_model()
    train_stats = load_stats()
except ImportError as e:
    # We'll show a warning later, after the page config
    import_warning = f"Direct backend import failed: {e}. Will try HTTP fallback."
except Exception as e:
    import_warning = f"Failed to load model: {e}"
    BACKEND_AVAILABLE = False

# ============================================================================
# 3. UI Setup
# ============================================================================
st.title("🏡 AI Real Estate Agent")
st.markdown("Describe a property in plain English, review the extracted features, and get an AI‑powered price prediction.")

# Show any import warnings now (after page config)
if 'import_warning' in locals():
    st.warning(import_warning)

# API fallback URL (only used if BACKEND_AVAILABLE is False)
API_URL = "http://localhost:8000"

# Session state initialization
if "extraction" not in st.session_state:
    st.session_state.extraction = None
if "edited_features" not in st.session_state:
    st.session_state.edited_features = {}

# ============================================================================
# 4. User Input
# ============================================================================
query = st.text_input(
    "Your property description:",
    placeholder="e.g., 3 bed, 2 bath ranch with a 2‑car garage, 1800 sqft, nice neighborhood"
)

col1, col2 = st.columns([1, 2])
with col1:
    prompt_version = st.selectbox("Prompt version:", ["v1", "v2"], index=0)

# ============================================================================
# 5. Feature Extraction
# ============================================================================
if st.button("Extract Features") and query:
    with st.spinner("Analyzing your description..."):
        if BACKEND_AVAILABLE:
            try:
                extraction_result = extract_features_direct(query, prompt_version)
                st.session_state.extraction = extraction_result.dict()
                st.session_state.edited_features = {
                    feat: data["value"] if data["value"] is not None else ""
                    for feat, data in st.session_state.extraction["features"].items()
                }
            except Exception as e:
                st.error(f"Extraction failed: {e}")
        else:
            try:
                resp = requests.post(
                    f"{API_URL}/extract",
                    json={"query": query, "prompt_version": prompt_version},
                    timeout=10
                )
                if resp.status_code == 200:
                    st.session_state.extraction = resp.json()
                    st.session_state.edited_features = {
                        feat: data["value"] if data["value"] is not None else ""
                        for feat, data in st.session_state.extraction["features"].items()
                    }
                else:
                    st.error(f"Extraction failed: {resp.text}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to backend API. Make sure FastAPI is running on localhost:8000.")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

# ============================================================================
# 6. Feature Editing Form
# ============================================================================
if st.session_state.extraction:
    st.subheader("Review & Edit Features")
    missing = st.session_state.extraction.get("missing_features", [])
    if missing:
        st.warning(f"⚠️ Missing features: {', '.join(missing)}. Please fill them below.")

    cols = st.columns(2)
    updated_features = {}
    for i, (feat, data) in enumerate(st.session_state.extraction["features"].items()):
        with cols[i % 2]:
            confidence = data["confidence"]
            reasoning = data.get("reasoning", "")
            if feat == "HouseStyle":
                options = ["1Story", "2Story", "1.5Fin", "1.5Unf", "SFoyer", "SLvl", "2.5Fin", "2.5Unf"]
                current_val = st.session_state.edited_features.get(feat, "")
                idx = options.index(current_val) if current_val in options else 0
                val = st.selectbox(f"{feat} ({confidence})", options, index=idx, help=reasoning)
            else:
                val = st.text_input(
                    f"{feat} ({confidence})",
                    value=str(st.session_state.edited_features.get(feat, "")),
                    help=reasoning
                )
            updated_features[feat] = val

    st.session_state.edited_features = updated_features

    # ========================================================================
    # 7. Prediction
    # ========================================================================
    if st.button("Predict Price", type="primary"):
        payload_features = {k: (v if v != "" else None) for k, v in st.session_state.edited_features.items()}

        with st.spinner("Calculating prediction..."):
            if BACKEND_AVAILABLE:
                try:
                    input_df = prepare_features_for_prediction(payload_features)
                    pred_log = model_pipeline.predict(input_df)[0]
                    pred_price = float(np.expm1(pred_log))

                    interpretation = generate_interpretation(
                        features=payload_features,
                        predicted_price=pred_price,
                        stats=train_stats
                    )

                    st.success(f"### Predicted Sale Price: ${pred_price:,.0f}")
                    st.info(interpretation)
                    with st.expander("Market Comparison"):
                        st.metric("Median Price", f"${train_stats['median']:,.0f}")
                        st.metric("Typical Range", f"${train_stats['q1']:,.0f} – ${train_stats['q3']:,.0f}")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
            else:
                try:
                    pred_resp = requests.post(
                        f"{API_URL}/predict",
                        json={"features": payload_features, "query": query},
                        timeout=10
                    )
                    if pred_resp.status_code == 200:
                        pred = pred_resp.json()
                        st.success(f"### Predicted Sale Price: ${pred['predicted_price']:,.0f}")
                        st.info(pred["interpretation"])
                        with st.expander("Market Comparison"):
                            stats = pred["comparison_stats"]
                            st.metric("Median Price", f"${stats['median']:,.0f}")
                            st.metric("Typical Range", f"${stats['q1']:,.0f} – ${stats['q3']:,.0f}")
                    else:
                        st.error(f"Prediction failed: {pred_resp.text}")
                except Exception as e:
                    st.error(f"Prediction request failed: {e}")