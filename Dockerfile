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

# Final code copy
COPY --chown=user app ./app

# Fix permissions and switch user
RUN chown -R user:user /app
USER user

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]