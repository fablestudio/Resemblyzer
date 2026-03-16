# RunPod Dockerfile for Resemblyzer Voice API
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Required at runtime (set via RunPod secrets or .env locally):
#   SUPABASE_URL            — e.g. {{ RUNPOD_SECRET_SUPABASE_URL }}
#   SUPABASE_SERVICE_KEY    — e.g. {{ RUNPOD_SECRET_SUPABASE_SERVICE_KEY }}

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

# Copy the RunPod handler and Supabase client
COPY rp_handler.py /app/rp_handler.py
COPY supabase_client.py /app/supabase_client.py

# Copy test input for local testing
COPY test_input.json /app/test_input.json

# Create cache directories
RUN mkdir -p /app/audio_cache /tmp/resemblyzer_audio

# Start the container with the RunPod handler
CMD ["python3", "-u", "rp_handler.py"]
