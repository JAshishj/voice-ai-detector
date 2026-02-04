FROM python:3.9-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
# Hugging Face Spaces expects port 7860
ENV PORT=7860

COPY requirements.txt .

# Install typing-extensions first from PyPi to avoid naming conflict on PyTorch index

# Install typing-extensions first from PyPi to avoid naming conflict on PyTorch index
RUN pip install --no-cache-dir typing-extensions

# Install specific CPU-only torch versions
RUN pip install --no-cache-dir "torch>=2.6.0" "torchaudio>=2.6.0" --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

# Download model from GitHub Release (as root to use apt-get)
RUN apt-get update && apt-get install -y curl && \
    mkdir -p model && \
    curl -L https://github.com/JAshishj/voice-ai-detector/releases/download/v1.0.0/detector.pt -o model/detector.pt && \
    apt-get remove -y curl && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# Pre-download base model files
COPY download_base.py .
RUN python download_base.py && rm download_base.py

# Set up non-root user and fix permissions
RUN useradd -m -u 1000 user && \
    chown -R user:user /app
USER user
ENV HOME=/home/user
ENV PATH=/home/user/.local/bin:$PATH

COPY --chown=user app ./app
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]