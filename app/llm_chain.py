# app/llm_chain.py
import json
import logging
from typing import Dict, Any
from pathlib import Path
from app import config
from app.pydan import FeatureExtractionOutput

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# 1. Load Prompt Templates
# -------------------------------------------------------------------
PROMPT_DIR = Path(__file__).parent.parent / "prompts"

def load_extraction_prompt(version: str) -> str:
    """Load the extraction prompt template for the given version."""
    file_path = PROMPT_DIR / f"extraction_{version}.txt"
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def load_interpretation_prompt() -> str:
    """Load the interpretation prompt template."""
    file_path = PROMPT_DIR / "interpretation.txt"
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# -------------------------------------------------------------------
# 2. LLM Provider Configuration
# -------------------------------------------------------------------
if config.LLM_PROVIDER == "groq":
    from groq import Groq
    client = Groq(api_key=config.GROQ_API_KEY)
    MODEL = config.GROQ_MODEL

    def call_llm(prompt: str) -> str:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()

else:
    raise ValueError(f"Unsupported LLM_PROVIDER: {config.LLM_PROVIDER}")

# -------------------------------------------------------------------
# 3. Core Functions
# -------------------------------------------------------------------
def extract_features(query: str, version: str = "v1") -> FeatureExtractionOutput:
    template = load_extraction_prompt(version)
    prompt = template.replace("{query}", query)  # Using replace to avoid brace issues

    content = call_llm(prompt)
    logger.info(f"Raw LLM response:\n{content}")

    # Clean markdown code blocks
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}")
        raise ValueError(f"LLM returned invalid JSON: {content[:200]}")

    # If LLM returned the full schema, use it directly
    if "features" in data and "missing_features" in data and "raw_query" in data:
        return FeatureExtractionOutput(**data)

    # Otherwise, assume data is just the features dict
    features_dict = data
    missing = [
        feat for feat, info in features_dict.items()
        if info.get("confidence") == "missing" or info.get("value") is None
    ]
    return FeatureExtractionOutput(
        features=features_dict,
        missing_features=missing,
        raw_query=query
    )

def generate_interpretation(
    features: Dict[str, Any],
    predicted_price: float,
    stats: Dict[str, Any]
) -> str:
    """
    Stage 2: Generate human‑readable interpretation of the prediction.
    """
    template = load_interpretation_prompt()
    prompt = template.format(
        features_dict=json.dumps(features, indent=2),
        predicted_price=predicted_price,
        median_price=stats["median"],
        q1=stats["q1"],
        q3=stats["q3"],
        top_features=", ".join(stats.get("top_features", ["OverallQual", "GrLivArea"]))
    )

    return call_llm(prompt)