from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class ExtractedFeature(BaseModel):
    value: Optional[Any] = None
    confidence: str  # "high", "medium", "low", "missing"
    reasoning: Optional[str] = None

class FeatureExtractionOutput(BaseModel):
    features: Dict[str, ExtractedFeature]
    missing_features: List[str]
    raw_query: str

class PredictionResponse(BaseModel):
    extracted_features: Dict[str, ExtractedFeature]
    predicted_price: float
    interpretation: str
    comparison_stats: Dict[str, float]   # median, q1, q3

