# ---- Console build ----
FROM node:20-slim AS console
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --no-audit --no-fund || npm install --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ---- API + console serving ----
FROM python:3.9-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
# Hugging Face Spaces expects port 7860
ENV PORT=7860
ENV NUMBA_CACHE_DIR=/tmp
ENV TORCH_THREADS=4
ENV QUANTIZE=1
ENV MODEL_VERSION=detector.pt

COPY requirements.txt .

# Install typing-extensions first from PyPi to avoid naming conflict on PyTorch index
RUN pip install --no-cache-dir typing-extensions

# Install specific CPU-only torch versions
RUN pip install --no-cache-dir "torch>=2.6.0" "torchaudio>=2.6.0" --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

# Install ffmpeg and libsndfile1 for audio processing
RUN apt-get update && apt-get install -y ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# Set up non-root user (Hugging Face requirement)
RUN useradd -m -u 1000 user
ENV HOME=/home/user
ENV PATH=/home/user/.local/bin:$PATH

# Copy local model folder (managed by Git LFS)
COPY --chown=user model ./model

# Pre-download base model files
COPY download_base.py .
RUN python download_base.py && rm download_base.py

# Final code copy (backend + built console)
COPY --chown=user app ./app
COPY --chown=user --from=console /build/dist ./frontend/dist

# Fix permissions and switch user
RUN chown -R user:user /app
USER user

EXPOSE 7860
HEALTHCHECK --interval=60s --timeout=10s --start-period=120s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health').read()" || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
