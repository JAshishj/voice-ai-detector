FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .

# Install specific CPU-only torch versions BEFORE requirements to save space (approx 4-5GB)
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY model ./model
EXPOSE 8000
CMD sh -c "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"