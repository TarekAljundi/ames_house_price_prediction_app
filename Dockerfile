FROM python:3.12-slim

WORKDIR /app

# Install system dependencies (if using xgboost)
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY model/ ./model/
COPY prompts/ ./prompts/

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Set environment variables (override at runtime)
# WARNING: Do not hardcode secrets here – pass via -e or --env-file
ENV GROQ_API_KEY=""
ENV LLM_PROVIDER="groq"
ENV GROQ_MODEL="llama-3.3-70b-versatile"

# Run both services
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8000 & streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0"]