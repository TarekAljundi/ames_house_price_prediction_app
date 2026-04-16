import streamlit as st
import requests
import json

API_URL = "http://localhost:8000"

st.set_page_config(page_title="AI Real Estate Agent", layout="wide")
st.title("🏡 AI Real Estate Agent")
st.markdown("Describe a property in plain English, review the extracted features, and get an AI‑powered price prediction.")

# Initialize session state
if "extraction" not in st.session_state:
    st.session_state.extraction = None
if "edited_features" not in st.session_state:
    st.session_state.edited_features = {}

# Input
query = st.text_input("Your property description:", placeholder="e.g., 3 bed, 2 bath ranch with a 2‑car garage, 1800 sqft, nice neighborhood")

col1, col2 = st.columns([1, 2])
with col1:
    prompt_version = st.selectbox("Prompt version:", ["v1", "v2"], index=0)

if st.button("Extract Features") and query:
    with st.spinner("Analyzing your description..."):
        resp = requests.post(f"{API_URL}/extract", json={"query": query, "prompt_version": prompt_version})
        if resp.status_code == 200:
            st.session_state.extraction = resp.json()
            # Initialize edited_features from extracted values
            st.session_state.edited_features = {
                feat: data["value"] if data["value"] is not None else ""
                for feat, data in st.session_state.extraction["features"].items()
            }
        else:
            st.error(f"Extraction failed: {resp.text}")

# Display editable form if extraction exists
if st.session_state.extraction:
    st.subheader("Review & Edit Features")
    missing = st.session_state.extraction.get("missing_features", [])
    if missing:
        st.warning(f"⚠️ Missing features: {', '.join(missing)}. Please fill them below.")

    # Create two columns for the form
    cols = st.columns(2)
    updated_features = {}
    for i, (feat, data) in enumerate(st.session_state.extraction["features"].items()):
        with cols[i % 2]:
            confidence = data["confidence"]
            reasoning = data.get("reasoning", "")
            # Determine input type
            if feat == "HouseStyle":
                options = ["1Story", "2Story", "1.5Fin", "1.5Unf", "SFoyer", "SLvl", "2.5Fin", "2.5Unf"]
                idx = options.index(st.session_state.edited_features[feat]) if st.session_state.edited_features[feat] in options else 0
                val = st.selectbox(f"{feat} ({confidence})", options, index=idx, help=reasoning)
            else:
                val = st.text_input(f"{feat} ({confidence})", value=st.session_state.edited_features[feat], help=reasoning)
            updated_features[feat] = val

    st.session_state.edited_features = updated_features

    if st.button("Predict Price", type="primary"):
        # Convert empty strings to None before sending
        payload_features = {k: (v if v != "" else None) for k, v in st.session_state.edited_features.items()}
        with st.spinner("Calculating prediction..."):
            pred_resp = requests.post(f"{API_URL}/predict", json={"features": payload_features, "query": query})
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