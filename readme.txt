
🏡 AI Real Estate Agent

    LLM Prompt Chaining + Supervised ML + Containerized Deployment

    A two‑stage AI system that extracts property features from plain English, predicts sale price using a trained XGBoost/RandomForest model, and generates human‑readable interpretations – all served via a FastAPI backend and Streamlit UI, deployable with Docker.

📌Features

    Natural Language Feature Extraction – Uses Groq's LLM (Llama 3.3) to parse user queries into structured features with confidence scores.

    Machine Learning Prediction – scikit‑learn pipeline with ColumnTransformer, trained on the Ames Housing dataset (12 key features).

    Interpretable Output – Second LLM stage explains why the price is high/low relative to market context.

    Editable UI – Review and correct extracted features before final prediction.

    Dockerized Deployment – Runs anywhere with a single docker run command.

    Prompt Versioning – Two extraction prompt variants compared and logged.

🏗️ Architecture

    User Query → LLM Stage 1 (Extract Features) → Pydantic Validation
                    ↓
            ML Model Prediction
                    ↓
            LLM Stage 2 (Interpretation) → Final Response
                    ↓
            FastAPI Backend ←→ Streamlit UI (Docker)

📂 Project Structure

    ames_house_price_prediction_app/
    ├── app/
    │   ├── __init__.py
    │   ├── main.py                 # FastAPI endpoints
    │   ├── streamlit_app.py        # UI frontend
    │   ├── llm_chain.py            # Prompt handling & Groq calls
    │   ├── pydan.py              # Pydantic validation models
    │   ├── utils.py                # Model loading & preprocessing
    │   └── config.py               # Environment configuration
    ├── model/
    │   ├── model_pipeline.joblib   # Trained sklearn pipeline
    │   └── train_stats.joblib      # Median, quartiles, feature importance
    ├── prompts/
    │   ├── extraction_v1.txt       # Explicit instruction prompt
    │   ├── extraction_v2.txt       # Few‑shot example prompt
    │   └── interpretation.txt      # Stage 2 interpretation prompt
    ├── notebooks/
    │   └── ml_pipeline_prompt_experiments.ipynb  # EDA, training
    ├── Dockerfile
    ├── requirements.txt
    ├── runtime.txt                 # Python 3.10 for Streamlit Cloud
    └── README.md

🚀 Quick Start

    1. Clone the Repository

        git clone https://github.com/TarekAljundi/ames_house_price_prediction_app.git

        cd ames_house_price_prediction_app

    2. Set Up Environment Variables

        Create a .env file in the project root (or set environment variables):

            LLM_PROVIDER=groq
            GROQ_API_KEY=your_groq_api_key_here
            GROQ_MODEL=llama-3.3-70b-versatile

    3. Run with Docker (Recommended)    

        docker build -t ai-realestate-agent .
        docker run -p 8000:8000 -p 8501:8501 --env-file .env ai-realestate-agent

        FastAPI backend: http://localhost:8000

        Streamlit UI: http://localhost:8501

    4. Run Locally (Development)

           # Install dependencies

            pip install -r requirements.txt

            # Terminal 1: FastAPI
            uvicorn app.main:app --reload --port 8000

            # Terminal 2: Streamlit
            streamlit run app/streamlit_app.py
🧪 Example Usage

            Query:

            "3 bedroom, 2.5 bath ranch with a 2‑car garage, 1800 square feet, nice neighborhood"

            Output:

            Extracted Features: OverallQual=7, GrLivArea=1800, GarageCars=2, HouseStyle=1Story, etc.

            Predicted Price: ~$245,000

            Interpretation: "The predicted price is above the market median, driven by the above‑average quality and desirable ranch style."    


📊 Dataset & Model

            Dataset: Ames Housing (1460 sales)

            Selected Features (12): OverallQual, GrLivArea, GarageArea, TotalBsmtSF, FullBath, 1stFlrSF, TotRmsAbvGrd, LotArea, GarageCars, MasVnrArea, LotFrontage, HouseStyle

            Target: SalePrice 

            Model: XGBoost / RandomForest (swappable) within a scikit‑learn Pipeline

            Validation: Three‑way split (train/val/test), no data leakage

🧰 Technologies Used

            LLM: Groq (Llama 3.3 70B)

            Backend: FastAPI, Uvicorn, Pydantic

            Frontend: Streamlit

            ML: scikit‑learn, XGBoost, pandas, numpy

            Deployment: Docker, Streamlit Cloud

            Notebooks: Google Colab

 👤 Author

    Tarek Aljundi
    GitHub: https://github.com/TarekAljundi

    