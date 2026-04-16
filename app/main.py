from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
from app.pydan import FeatureExtractionOutput, PredictionResponse
from app.llm_chain import extract_features, generate_interpretation
from app.utils import load_model, load_stats, prepare_features_for_prediction

app = FastAPI(title="AI Real Estate Agent")

# Load model and training statistics once at startup
model_pipeline = load_model()
train_stats = load_stats()

class QueryRequest(BaseModel):
    query: str
    prompt_version: str = "v1"

class ManualFeatureRequest(BaseModel):
    features: dict          # user‑edited features (after extraction)
    query: str

@app.post("/extract", response_model=FeatureExtractionOutput)
async def extract(request: QueryRequest):
    """Stage 1: Extract features from natural language query."""
    try:
        result = extract_features(request.query, request.prompt_version)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: ManualFeatureRequest):
    """Stage 2: Predict price and generate interpretation."""
    try:
        # Prepare input DataFrame for the model pipeline
        input_df = prepare_features_for_prediction(request.features)
        # Predict log price and transform back
        pred= model_pipeline.predict(input_df)[0]
        pred_price = float(pred)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

    try:
        interpretation = generate_interpretation(
            features=request.features,
            predicted_price=pred_price,
            stats=train_stats
        )
    except Exception as e:
        interpretation = "Interpretation unavailable at this time."

    # Build response with extraction metadata (we can store it in session, but here we use empty dict)
    return PredictionResponse(
        extracted_features={},  # In a real app, pass from session/state
        predicted_price=pred_price,
        interpretation=interpretation,
        comparison_stats={
            "median": train_stats["median"],
            "q1": train_stats["q1"],
            "q3": train_stats["q3"]
        }
    )