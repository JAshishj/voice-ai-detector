FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .

# Install typing-extensions first from PyPi to avoid naming conflict on PyTorch index
RUN pip install --no-cache-dir typing-extensions

# Install specific CPU-only torch versions
# explicitly requesting >=2.6.0 due to CVE-2025-32434 requirements from transformers
RUN pip install --no-cache-dir "torch>=2.6.0" "torchaudio>=2.6.0" --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

# Download model from GitHub Release to bypass Git LFS pointer issues
RUN apt-get update && apt-get install -y curl && \
    mkdir -p model && \
    curl -L https://github.com/JAshishj/voice-ai-detector/releases/download/v1.0.0/detector.pt -o model/detector.pt && \
    apt-get remove -y curl && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# Pre-download base model files to avoid runtime download and timeouts
COPY download_base.py .
RUN python download_base.py && rm download_base.py

COPY app ./app
EXPOSE 8000
CMD sh -c "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"