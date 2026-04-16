import joblib
import pandas as pd
from app.config import MODEL_PATH, STATS_PATH

# Define the exact feature order expected by the pipeline
FEATURE_NAMES = [
    'OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF', 'FullBath',
    '1stFlrSF', 'TotRmsAbvGrd', 'LotArea', 'GarageCars', 'MasVnrArea',
    'LotFrontage', 'HouseStyle'
]

def load_model():
    """Load serialized sklearn pipeline."""
    return joblib.load(MODEL_PATH)

def load_stats():
    """Load training statistics (median, quartiles, top features)."""
    return joblib.load(STATS_PATH)

def prepare_features_for_prediction(features_dict: dict) -> pd.DataFrame:
    """
    Convert the extracted feature dict into a DataFrame with correct column order.
    Handles missing values by filling with median (from training) or 'Missing' for categorical.
    """
    # Start with a dict of default values (median/mode from training)
    # For simplicity, we use typical fallbacks; in practice, load from train_stats
    defaults = {
        'OverallQual': 5,
        'GrLivArea': 1500,
        'GarageArea': 480,
        'TotalBsmtSF': 1000,
        'FullBath': 2,
        '1stFlrSF': 1200,
        'TotRmsAbvGrd': 6,
        'LotArea': 10000,
        'GarageCars': 2,
        'MasVnrArea': 0,
        'LotFrontage': 60,
        'HouseStyle': '1Story'
    }

    row = {}
    for feat in FEATURE_NAMES:
        val = features_dict.get(feat)
        if val is None or val == "":
            row[feat] = defaults[feat]
        else:
            # Convert numeric strings to int/float
            if feat != 'HouseStyle':
                try:
                    row[feat] = float(val) if '.' in str(val) else int(val)
                except:
                    row[feat] = defaults[feat]
            else:
                row[feat] = str(val)

    return pd.DataFrame([row])[FEATURE_NAMES]


