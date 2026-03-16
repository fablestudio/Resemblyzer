import runpod
import logging
import hashlib
import os
import base64
import tempfile
import requests
import numpy as np
from pathlib import Path

from resemblyzer import VoiceEncoder, preprocess_wav

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Global model instance
encoder = None

AUDIO_CACHE_DIR = Path("/app/audio_cache")


def get_audio_cache_hash(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def get_cached_filename(url: str, ext: str) -> str:
    name = url.split("/")[-1].split(".")[0][:20]
    name = "".join(c for c in name if c.isalnum() or c in "-_")
    url_hash = get_audio_cache_hash(url)
    return f"{name}_{url_hash}{ext}"


def download_cached_audio(audio_url: str) -> str:
    """Download audio file from URL with caching support."""
    AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Check cache
    for ext in [".wav", ".mp3", ".m4a", ".flac"]:
        cached_file = AUDIO_CACHE_DIR / get_cached_filename(audio_url, ext)
        if cached_file.exists():
            logger.info(f"Cache hit: {cached_file}")
            return str(cached_file)

    logger.info(f"Downloading audio from URL...")
    response = requests.get(audio_url, stream=True)
    response.raise_for_status()

    # Detect extension
    file_extension = ".wav"
    url_lower = audio_url.lower()
    for ext in [".mp3", ".wav", ".m4a", ".flac"]:
        if url_lower.endswith(ext):
            file_extension = ext
            break
    else:
        content_type = response.headers.get("content-type", "").lower()
        if "mp3" in content_type or "mpeg" in content_type:
            file_extension = ".mp3"
        elif "flac" in content_type:
            file_extension = ".flac"

    cached_file = AUDIO_CACHE_DIR / get_cached_filename(audio_url, file_extension)
    with open(cached_file, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    logger.info(f"Downloaded and cached: {cached_file}")
    return str(cached_file)


def decode_base64_audio(audio_base64: str, filename: str = "input.wav") -> str:
    """Decode base64 audio data to a temporary file."""
    tmp_dir = Path("/tmp/resemblyzer_audio")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / filename
    audio_bytes = base64.b64decode(audio_base64)
    with open(tmp_path, "wb") as f:
        f.write(audio_bytes)
    return str(tmp_path)


def resolve_audio(audio_spec: dict, key_prefix: str = "audio") -> str:
    """
    Resolve an audio source from either a URL or base64 data.
    Looks for keys: {key_prefix}_url or {key_prefix}_base64
    Returns path to the audio file.
    """
    url_key = f"{key_prefix}_url"
    b64_key = f"{key_prefix}_base64"

    if url_key in audio_spec and audio_spec[url_key]:
        return download_cached_audio(audio_spec[url_key])
    elif b64_key in audio_spec and audio_spec[b64_key]:
        return decode_base64_audio(audio_spec[b64_key], f"{key_prefix}.wav")
    else:
        raise ValueError(f"No audio source provided. Expected '{url_key}' or '{b64_key}'.")


def handler(event):
    """
    RunPod serverless handler for Resemblyzer speaker identification.

    Supports two modes:

    1. "identify" - Compare an input audio against sample speaker audios
       Input:
       {
         "mode": "identify",
         "input_audio_url": "https://...",        # OR "input_audio_base64": "<base64>"
         "speakers": {
           "speaker_name": {
             "samples": [
               {"audio_url": "https://..."},      # OR {"audio_base64": "<base64>"}
               ...
             ]
           },
           ...
         }
       }

    2. "embed" - Generate voice embedding(s) for audio file(s)
       Input:
       {
         "mode": "embed",
         "audio_url": "https://...",              # OR "audio_base64": "<base64>"
         # OR for multiple files:
         "audio_files": [
           {"audio_url": "https://..."},
           {"audio_base64": "<base64>"}
         ]
       }

    3. "compare" - Compare two audio files directly
       Input:
       {
         "mode": "compare",
         "audio_a_url": "https://...",            # OR "audio_a_base64": "<base64>"
         "audio_b_url": "https://...",            # OR "audio_b_base64": "<base64>"
       }
    """
    global encoder

    input_data = event["input"]
    mode = input_data.get("mode", "identify")

    logger.info(f"Processing request, mode: {mode}")

    try:
        if mode == "identify":
            return handle_identify(input_data)
        elif mode == "embed":
            return handle_embed(input_data)
        elif mode == "compare":
            return handle_compare(input_data)
        else:
            return {"status": "error", "message": f"Unknown mode: {mode}. Use 'identify', 'embed', or 'compare'."}

    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


def handle_identify(input_data: dict) -> dict:
    """Compare input audio against named speaker samples."""
    global encoder

    # Resolve input audio
    input_path = resolve_audio(input_data, "input_audio")
    input_wav = preprocess_wav(input_path)
    input_embed = encoder.embed_utterance(input_wav)

    # Process each speaker's samples
    speakers = input_data.get("speakers", {})
    if not speakers:
        return {"status": "error", "message": "No speakers provided."}

    results = {}
    for speaker_name, speaker_data in speakers.items():
        samples = speaker_data.get("samples", [])
        if not samples:
            logger.warning(f"No samples for speaker: {speaker_name}")
            continue

        # Embed each sample and average
        sample_wavs = []
        for sample in samples:
            sample_path = resolve_audio(sample, "audio")
            sample_wav = preprocess_wav(sample_path)
            sample_wavs.append(sample_wav)

        speaker_embed = encoder.embed_speaker(sample_wavs)
        similarity = float(np.dot(input_embed, speaker_embed))
        results[speaker_name] = {
            "similarity": round(similarity, 4),
            "num_samples": len(sample_wavs),
        }

    # Sort by similarity descending
    sorted_results = dict(sorted(results.items(), key=lambda x: x[1]["similarity"], reverse=True))

    # Best match
    best_match = next(iter(sorted_results)) if sorted_results else None
    best_score = sorted_results[best_match]["similarity"] if best_match else 0.0

    return {
        "status": "success",
        "best_match": best_match,
        "best_score": round(best_score, 4),
        "results": sorted_results,
    }


def handle_embed(input_data: dict) -> dict:
    """Generate voice embedding(s) for one or more audio files."""
    global encoder

    embeddings = []

    # Multiple files
    audio_files = input_data.get("audio_files", [])
    if audio_files:
        for i, audio_spec in enumerate(audio_files):
            path = resolve_audio(audio_spec, "audio")
            wav = preprocess_wav(path)
            embed = encoder.embed_utterance(wav)
            embeddings.append({
                "index": i,
                "embedding": embed.tolist(),
            })
    else:
        # Single file
        path = resolve_audio(input_data, "audio")
        wav = preprocess_wav(path)
        embed = encoder.embed_utterance(wav)
        embeddings.append({
            "index": 0,
            "embedding": embed.tolist(),
        })

    return {
        "status": "success",
        "embeddings": embeddings,
        "embedding_size": 256,
    }


def handle_compare(input_data: dict) -> dict:
    """Compare two audio files and return similarity."""
    global encoder

    path_a = resolve_audio(input_data, "audio_a")
    path_b = resolve_audio(input_data, "audio_b")

    wav_a = preprocess_wav(path_a)
    wav_b = preprocess_wav(path_b)

    embed_a = encoder.embed_utterance(wav_a)
    embed_b = encoder.embed_utterance(wav_b)

    similarity = float(np.dot(embed_a, embed_b))

    return {
        "status": "success",
        "similarity": round(similarity, 4),
    }


def initialize_model():
    """Initialize the VoiceEncoder model."""
    global encoder
    try:
        logger.info("Initializing VoiceEncoder model...")

        # Check for custom weights path
        weights_path = os.environ.get("RESEMBLYZER_WEIGHTS_PATH")
        if weights_path and Path(weights_path).exists():
            logger.info(f"Loading custom weights from: {weights_path}")
            encoder = VoiceEncoder(weights_fpath=weights_path)
        else:
            encoder = VoiceEncoder()

        logger.info("VoiceEncoder initialized successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize VoiceEncoder: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    if initialize_model():
        logger.info("Starting RunPod serverless handler...")
        runpod.serverless.start({"handler": handler})
    else:
        logger.error("Failed to initialize model. Exiting.")
        exit(1)
