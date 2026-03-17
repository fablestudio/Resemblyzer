# RunPod Dockerfile for Resemblyzer Voice API
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies (webrtcvad needs build tools, librosa needs ffmpeg/libsndfile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements-runpod.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements-runpod.txt

# Copy the resemblyzer package (includes pretrained.pt model)
COPY resemblyzer/ /app/resemblyzer/

# Copy the RunPod handler
COPY rp_handler.py /app/rp_handler.py

# Create cache directories
RUN mkdir -p /app/audio_cache /tmp/resemblyzer_audio

# Start the container with the RunPod handler
CMD ["python3", "-u", "rp_handler.py"]
