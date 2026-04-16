import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")

# OpenAI
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

# Model paths
MODEL_PATH = BASE_DIR / "model" / "model_pipeline.joblib"
STATS_PATH = BASE_DIR / "model" / "train_stats.joblib"

# Prompts directory
PROMPT_DIR = BASE_DIR / "prompts"