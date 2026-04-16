import streamlit as st
import requests
import json
import sys
from pathlib import Path

# Add app directory to path so we can import local modules
sys.path.append(str(Path(__file__).parent))

# Attempt to import FastAPI backend functions directly
try:
    from llm_chain import extract_features as extract_features_direct, generate_interpretation
    from utils import load_model, load_stats, prepare_features_for_prediction
    import numpy as np
    BACKEND_AVAILABLE = True
except ImportError as e:
    BACKEND_AVAILABLE = False
    st.warning(f"Direct backend import failed: {e}. Will try HTTP fallback.")

# Configuration
API_URL = "http://localhost:8000"
USE_HTTP_API = False  # Set to True only if you want to force HTTP even if imports work

st.set_page_config(page_title="AI Real Estate Agent", layout="wide")
st.title("🏡 AI Real Estate Agent")
st.markdown("Describe a property in plain English, review the extracted features, and get an AI‑powered price prediction.")

# Initialize session state
if "extraction" not in st.session_state:
    st.session_state.extraction = None
if "edited_features" not in st.session_state:
    st.session_state.edited_features = {}

# Load model and stats once (if using direct mode)
if BACKEND_AVAILABLE and not USE_HTTP_API:
    try:
        model_pipeline = load_model()
        train_stats = load_stats()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

# Input
query = st.text_input("Your property description:", 
                      placeholder="e.g., 3 bed, 2 bath ranch with a 2‑car garage, 1800 sqft, nice neighborhood")
col1, col2 = st.columns([1, 2])
with col1:
    prompt_version = st.selectbox("Prompt version:", ["v1", "v2"], index=0)

if st.button("Extract Features") and query:
    with st.spinner("Analyzing your description..."):
        if BACKEND_AVAILABLE and not USE_HTTP_API:
            # Direct function call
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
            # HTTP fallback
            try:
                resp = requests.post(f"{API_URL}/extract", 
                                     json={"query": query, "prompt_version": prompt_version},
                                     timeout=10)
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

# Display editable form if extraction exists
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
                val = st.text_input(f"{feat} ({confidence})", 
                                    value=str(st.session_state.edited_features.get(feat, "")),
                                    help=reasoning)
            updated_features[feat] = val

    st.session_state.edited_features = updated_features

    if st.button("Predict Price", type="primary"):
        payload_features = {k: (v if v != "" else None) for k, v in st.session_state.edited_features.items()}
        
        with st.spinner("Calculating prediction..."):
            if BACKEND_AVAILABLE and not USE_HTTP_API:
                # Direct prediction
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
                # HTTP fallback
                try:
                    pred_resp = requests.post(f"{API_URL}/predict", 
                                              json={"features": payload_features, "query": query},
                                              timeout=10)
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