FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY models/ ./models/
COPY prompts/ ./prompts/

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Set environment variables (override at runtime)
ENV OPENAI_API_KEY=""

# Run both services (use a process manager in production)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8000 & streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0"]