"""
RunPod serverless handler for Resemblyzer — stateless voice audio processing.

Endpoints (routed via "endpoint" field in input):

  POST /v1/embeddings       — Generate voice embeddings from audio
  POST /v1/audio/identify   — Speaker identification using provided reference embeddings or audio URLs
"""

import runpod
import logging
import hashlib
import os
import re
import base64
import numpy as np
from pathlib import Path

from resemblyzer import VoiceEncoder, preprocess_wav
from resemblyzer.hparams import sampling_rate

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global model
# ---------------------------------------------------------------------------
encoder = None

MODEL_ID = "resemblyzer-v1"
EMBEDDING_DIM = 256
AUDIO_CACHE_DIR = Path("/app/audio_cache")
DOWNLOAD_CONNECT_TIMEOUT_SECONDS = 5
DOWNLOAD_READ_TIMEOUT_SECONDS = 60
MAX_AUDIO_DOWNLOAD_BYTES = 50 * 1024 * 1024

# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def get_audio_cache_hash(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def get_cached_filename(url: str, ext: str) -> str:
    name = url.split("/")[-1].split(".")[0][:20]
    name = "".join(c for c in name if c.isalnum() or c in "-_")
    url_hash = get_audio_cache_hash(url)
    return f"{name}_{url_hash}{ext}"


def download_cached_audio(audio_url: str) -> str:
    import requests

    AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached_file = None

    for ext in [".wav", ".mp3", ".m4a", ".flac"]:
        cached_file = AUDIO_CACHE_DIR / get_cached_filename(audio_url, ext)
        if cached_file.exists():
            logger.info(f"Cache hit: {cached_file}")
            return str(cached_file)

    logger.info("Downloading audio from URL...")
    try:
        with requests.get(
            audio_url,
            stream=True,
            timeout=(
                DOWNLOAD_CONNECT_TIMEOUT_SECONDS,
                DOWNLOAD_READ_TIMEOUT_SECONDS,
            ),
        ) as response:
            response.raise_for_status()

            content_length = response.headers.get("content-length")
            if content_length and int(content_length) > MAX_AUDIO_DOWNLOAD_BYTES:
                raise ValueError(
                    f"Audio download exceeds the {MAX_AUDIO_DOWNLOAD_BYTES} byte limit"
                )

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
            bytes_downloaded = 0
            with open(cached_file, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if not chunk:
                        continue
                    bytes_downloaded += len(chunk)
                    if bytes_downloaded > MAX_AUDIO_DOWNLOAD_BYTES:
                        raise ValueError(
                            f"Audio download exceeds the {MAX_AUDIO_DOWNLOAD_BYTES} byte limit"
                        )
                    f.write(chunk)
    except requests.Timeout as exc:
        if cached_file and cached_file.exists():
            cached_file.unlink(missing_ok=True)
        raise ValueError(f"Timed out downloading audio from URL: {audio_url}") from exc
    except requests.RequestException as exc:
        if cached_file and cached_file.exists():
            cached_file.unlink(missing_ok=True)
        raise ValueError(f"Failed to download audio from URL: {audio_url}") from exc
    except ValueError:
        if cached_file and cached_file.exists():
            cached_file.unlink(missing_ok=True)
        raise

    logger.info(f"Downloaded and cached: {cached_file}")
    return str(cached_file)


def decode_base64_audio(audio_base64: str, filename: str = "input.wav") -> str:
    tmp_dir = Path("/tmp/resemblyzer_audio")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / filename
    audio_bytes = base64.b64decode(audio_base64)
    with open(tmp_path, "wb") as f:
        f.write(audio_bytes)
    return str(tmp_path)


def resolve_audio(spec: dict, key_prefix: str = "audio") -> str:
    """Resolve audio from {prefix}_url, {prefix}_base64, or {prefix}_path keys."""
    url_key = f"{key_prefix}_url"
    b64_key = f"{key_prefix}_base64"
    path_key = f"{key_prefix}_path"

    if path_key in spec and spec[path_key]:
        p = spec[path_key]
        if not Path(p).exists():
            raise ValueError(f"Local file not found: {p}")
        return p
    elif url_key in spec and spec[url_key]:
        return resolve_audio_url(spec[url_key])
    elif b64_key in spec and spec[b64_key]:
        return decode_base64_audio(spec[b64_key], f"{key_prefix}.wav")
    else:
        raise ValueError(f"No audio source provided. Expected '{url_key}', '{b64_key}', or '{path_key}'.")


def resolve_audio_url(url: str) -> str:
    """Resolve an audio URL (or local path) and return local file path."""
    if url.startswith("/") or url.startswith("file://"):
        local_path = url.replace("file://", "")
        if not Path(local_path).exists():
            raise ValueError(f"Local file not found: {local_path}")
        return local_path
    return download_cached_audio(url)


# ---------------------------------------------------------------------------
# WebVTT parsing
# ---------------------------------------------------------------------------


def parse_webvtt_timestamp(ts: str) -> float:
    parts = ts.strip().split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return float(parts[0])


def parse_webvtt(webvtt: str) -> list:
    segments = []
    blocks = re.split(r"\n\s*\n", webvtt.strip())

    for block in blocks:
        lines = block.strip().split("\n")
        if not lines:
            continue
        if lines[0].startswith("WEBVTT"):
            continue

        ts_line = None
        ts_idx = None
        for i, line in enumerate(lines):
            if "-->" in line:
                ts_line = line
                ts_idx = i
                break
        if ts_line is None:
            continue

        match = re.match(r"([\d:.]+)\s*-->\s*([\d:.]+)", ts_line)
        if not match:
            continue
        start = parse_webvtt_timestamp(match.group(1))
        end = parse_webvtt_timestamp(match.group(2))

        cue_index = None
        if ts_idx > 0 and lines[ts_idx - 1].strip().isdigit():
            cue_index = int(lines[ts_idx - 1].strip())

        text_lines = lines[ts_idx + 1:]
        text = " ".join(text_lines).strip()

        speaker = None
        voice_match = re.match(r"<v\s+([^>]+)>(.*)$", text)
        if voice_match:
            speaker = voice_match.group(1).strip()
            text = voice_match.group(2).strip()

        segments.append({
            "index": cue_index if cue_index is not None else len(segments),
            "start": round(start, 3),
            "end": round(end, 3),
            "text": text,
            "speaker": speaker,
        })

    return segments


def slice_wav_segment(wav: np.ndarray, start: float, end: float) -> np.ndarray:
    start_sample = max(0, int(start * sampling_rate))
    end_sample = min(len(wav), int(end * sampling_rate))
    return wav[start_sample:end_sample]


# ---------------------------------------------------------------------------
# Embedding generation
# ---------------------------------------------------------------------------


def generate_embedding(audio_path: str) -> np.ndarray:
    """Generate a 256-dim voice embedding from an audio file path."""
    wav = preprocess_wav(audio_path)
    return encoder.embed_utterance(wav)


# ---------------------------------------------------------------------------
# Endpoint: POST /v1/embeddings
# ---------------------------------------------------------------------------


def handle_embeddings_create(input_data: dict) -> dict:
    """
    Generate voice embeddings from audio. Pure compute — no DB interaction.

    Input:
      audio_url / audio_base64 / audio_path: audio source
      audio_files: list of {audio_url/audio_base64/audio_path} for batch

    Response (OpenAI-compatible):
      { object: "list", data: [{object: "embedding", index, embedding}], model, usage }
    """
    audio_files = input_data.get("audio_files", [])
    if not audio_files:
        audio_files = [input_data]

    results = []
    for i, item in enumerate(audio_files):
        audio_path = resolve_audio(item, "audio")
        embed = generate_embedding(audio_path)

        results.append({
            "object": "embedding",
            "index": i,
            "embedding": embed.tolist(),
        })

    return {
        "object": "list",
        "data": results,
        "model": MODEL_ID,
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }


# ---------------------------------------------------------------------------
# Endpoint: POST /v1/audio/identify
# ---------------------------------------------------------------------------


def handle_audio_identify(input_data: dict) -> dict:
    """
    Speaker identification with per-segment confidence scores.

    Reference speakers can be provided as:
      - Pre-computed embeddings: reference_embeddings: {voice_id: [float...], ...}
      - Audio URLs to embed on-the-fly: reference_audio_urls: {voice_id: "https://...", ...}
      - Mix of both (embeddings take precedence)

    Input:
      audio_url / audio_base64: input audio to analyze
      reference_embeddings: dict {voice_id: [256 floats]} — pre-computed embeddings
      reference_audio_urls: dict {voice_id: "url"} — audio to embed on-the-fly
      top_k: int (default 5)
      segmentation: "whole" | "auto" | {rate, resolution, ...} | webvtt string | list of segments

    Response:
      { object, model, duration, segments, speaker_summary, generated_embeddings? }
    """
    # Resolve input audio
    input_path = resolve_audio(input_data, "audio")
    input_wav = preprocess_wav(input_path)
    duration = len(input_wav) / sampling_rate

    # Build speaker embeddings from provided data
    ref_embeddings = input_data.get("reference_embeddings", {})
    ref_audio_urls = input_data.get("reference_audio_urls", {})

    if not ref_embeddings and not ref_audio_urls:
        raise ValueError(
            "Provide reference_embeddings and/or reference_audio_urls with at least one speaker"
        )

    top_k = input_data.get("top_k", 5)

    speaker_embeds = {}
    generated_embeddings = {}

    # Load pre-computed embeddings
    for vid, emb in ref_embeddings.items():
        speaker_embeds[vid] = np.array(emb, dtype=np.float32)

    # Generate embeddings from audio URLs (only for voices not already provided)
    for vid, url in ref_audio_urls.items():
        if vid in speaker_embeds:
            continue
        audio_path = resolve_audio_url(url)
        embed = generate_embedding(audio_path)
        speaker_embeds[vid] = embed
        # Return newly generated embeddings so the caller can cache them
        generated_embeddings[vid] = embed.tolist()

    if not speaker_embeds:
        raise ValueError("No valid speaker references could be resolved")

    # Determine segmentation mode
    segmentation = input_data.get("segmentation", "auto")

    if segmentation == "whole":
        result = _identify_whole(input_wav, speaker_embeds, top_k, duration)
    elif isinstance(segmentation, str) and segmentation.strip().startswith("WEBVTT"):
        segments = parse_webvtt(segmentation)
        result = _identify_segmented(input_wav, speaker_embeds, segments, top_k, duration)
    elif isinstance(segmentation, list):
        result = _identify_segmented(input_wav, speaker_embeds, segmentation, top_k, duration)
    else:
        diarize_config = input_data
        if isinstance(segmentation, dict):
            diarize_config = {**input_data, **segmentation}
        result = _identify_diarize(input_wav, speaker_embeds, diarize_config, top_k, duration)

    # Attach any newly generated embeddings so the caller can store them
    if generated_embeddings:
        result["generated_embeddings"] = generated_embeddings

    return result


def _score_top_k(scores: dict, top_k: int) -> list[dict]:
    """Return sorted top_k speakers with scores."""
    sorted_speakers = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [
        {"voice_id": vid, "score": round(s, 4)}
        for vid, s in sorted_speakers[:top_k]
    ]


def _identify_whole(input_wav, speaker_embeds, top_k, duration) -> dict:
    """Score the entire audio as a single segment against all speakers."""
    embed = encoder.embed_utterance(input_wav)
    scores = {vid: float(np.dot(embed, spk)) for vid, spk in speaker_embeds.items()}

    return {
        "object": "speaker_identification",
        "model": MODEL_ID,
        "duration": round(duration, 3),
        "segments": [{
            "index": 0,
            "start": 0.0,
            "end": round(duration, 3),
            "duration": round(duration, 3),
            "top_speakers": _score_top_k(scores, top_k),
            "scores": {vid: round(s, 4) for vid, s in scores.items()},
        }],
        "speaker_summary": _build_summary({vid: [scores[vid]] for vid in scores}),
    }


def _identify_segmented(input_wav, speaker_embeds, segments, top_k, duration) -> dict:
    """Score each provided segment independently against speakers."""
    logger.info(f"Processing {len(segments)} segments")
    segment_results = []
    all_scores = {vid: [] for vid in speaker_embeds}

    for seg in segments:
        start = float(seg["start"])
        end = float(seg["end"])
        seg_wav = slice_wav_segment(input_wav, start, end)

        min_samples = int(0.5 * sampling_rate)
        if len(seg_wav) < min_samples:
            logger.warning(f"Segment {seg.get('index', '?')} too short, skipping")
            segment_results.append({
                "index": seg.get("index"),
                "start": start,
                "end": end,
                "duration": round(end - start, 3),
                "text": seg.get("text", ""),
                "labeled_speaker": seg.get("speaker"),
                "top_speakers": [],
                "scores": {vid: 0.0 for vid in speaker_embeds},
                "skipped": True,
            })
            continue

        seg_embed = encoder.embed_utterance(seg_wav)
        scores = {}
        for vid, spk_embed in speaker_embeds.items():
            s = float(np.dot(seg_embed, spk_embed))
            scores[vid] = s
            all_scores[vid].append(s)

        segment_results.append({
            "index": seg.get("index"),
            "start": start,
            "end": end,
            "duration": round(end - start, 3),
            "text": seg.get("text", ""),
            "labeled_speaker": seg.get("speaker"),
            "top_speakers": _score_top_k(scores, top_k),
            "scores": {vid: round(s, 4) for vid, s in scores.items()},
        })

    return {
        "object": "speaker_identification",
        "model": MODEL_ID,
        "duration": round(duration, 3),
        "num_segments": len(segment_results),
        "segments": segment_results,
        "speaker_summary": _build_summary(all_scores),
    }


def _identify_diarize(input_wav, speaker_embeds, input_data, top_k, duration) -> dict:
    """Automatic sliding-window diarization."""
    rate = input_data.get("rate", 4)
    threshold_confident = input_data.get("threshold_confident", 0.75)
    threshold_uncertain = input_data.get("threshold_uncertain", 0.65)
    resolution = input_data.get("resolution", 0.5)

    logger.info(f"Diarizing {duration:.1f}s of audio at rate={rate}")

    _, cont_embeds, wav_splits = encoder.embed_utterance(
        input_wav, return_partials=True, rate=rate
    )

    times = np.array([((s.start + s.stop) / 2) / sampling_rate for s in wav_splits])
    speaker_names = list(speaker_embeds.keys())
    similarity_dict = {
        vid: cont_embeds @ embed for vid, embed in speaker_embeds.items()
    }

    # Build per-bucket speaker assignments
    total_dur = times[-1]
    buckets = []
    for t_start in np.arange(0, total_dur, resolution):
        t_end = t_start + resolution
        mask = (times >= t_start) & (times < t_end)
        if not mask.any():
            continue
        avg_sims = {vid: float(similarity_dict[vid][mask].mean()) for vid in speaker_names}
        best = max(avg_sims, key=avg_sims.get)
        score = avg_sims[best]

        if score > threshold_confident:
            confidence = "high"
        elif score > threshold_uncertain:
            confidence = "medium"
        else:
            confidence = "low"
            best = None

        buckets.append({
            "start": round(float(t_start), 3),
            "end": round(float(t_end), 3),
            "speaker": best,
            "score": round(score, 4),
            "confidence": confidence,
            "scores": {vid: round(v, 4) for vid, v in avg_sims.items()},
        })

    # Merge adjacent buckets with the same speaker
    segments = []
    if buckets:
        current = {
            "start": buckets[0]["start"],
            "end": buckets[0]["end"],
            "speaker": buckets[0]["speaker"],
            "scores_sum": {vid: buckets[0]["scores"][vid] for vid in speaker_names},
            "bucket_count": 1,
        }

        for b in buckets[1:]:
            if b["speaker"] == current["speaker"] and b["speaker"] is not None:
                current["end"] = b["end"]
                for vid in speaker_names:
                    current["scores_sum"][vid] += b["scores"][vid]
                current["bucket_count"] += 1
            else:
                segments.append(_finalize_segment(
                    current, speaker_names, threshold_confident, threshold_uncertain, top_k
                ))
                current = {
                    "start": b["start"],
                    "end": b["end"],
                    "speaker": b["speaker"],
                    "scores_sum": {vid: b["scores"][vid] for vid in speaker_names},
                    "bucket_count": 1,
                }

        segments.append(_finalize_segment(
            current, speaker_names, threshold_confident, threshold_uncertain, top_k
        ))

    # Per-speaker summary
    all_scores = {}
    for vid in speaker_names:
        sims = similarity_dict[vid]
        all_scores[vid] = sims.tolist()

    return {
        "object": "speaker_identification",
        "model": MODEL_ID,
        "duration": round(duration, 3),
        "num_segments": len(segments),
        "segments": segments,
        "speaker_summary": _build_summary(all_scores),
    }


def _finalize_segment(current, speaker_names, threshold_confident, threshold_uncertain, top_k):
    n = current["bucket_count"]
    avg_scores = {vid: round(current["scores_sum"][vid] / n, 4) for vid in speaker_names}
    best_score = avg_scores[current["speaker"]] if current["speaker"] else 0.0

    if best_score > threshold_confident:
        confidence = "high"
    elif best_score > threshold_uncertain:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "start": current["start"],
        "end": current["end"],
        "duration": round(current["end"] - current["start"], 3),
        "speaker": current["speaker"],
        "confidence": confidence,
        "top_speakers": _score_top_k(avg_scores, top_k),
        "scores": avg_scores,
    }


def _build_summary(all_scores: dict) -> dict:
    """Build per-speaker summary stats from collected scores."""
    summary = {}
    for vid, scores_list in all_scores.items():
        if not scores_list:
            summary[vid] = {"min": 0.0, "max": 0.0, "mean": 0.0}
            continue
        arr = np.array(scores_list)
        summary[vid] = {
            "min": round(float(arr.min()), 4),
            "max": round(float(arr.max()), 4),
            "mean": round(float(arr.mean()), 4),
        }
    return summary


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

ENDPOINT_HANDLERS = {
    "/v1/embeddings": handle_embeddings_create,
    "/v1/audio/identify": handle_audio_identify,
}


def handler(event):
    """RunPod serverless handler — routes to endpoint handlers."""
    global encoder

    input_data = event["input"]
    endpoint = input_data.get("endpoint", "/v1/audio/identify")

    logger.info(f"Request: endpoint={endpoint}")

    try:
        handler_fn = ENDPOINT_HANDLERS.get(endpoint)
        if not handler_fn:
            return {
                "error": {
                    "message": f"Unknown endpoint: {endpoint}. "
                    f"Available: {', '.join(ENDPOINT_HANDLERS.keys())}",
                    "type": "invalid_request_error",
                },
            }

        return handler_fn(input_data)

    except ValueError as e:
        return {
            "error": {
                "message": str(e),
                "type": "invalid_request_error",
            },
        }
    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        return {
            "error": {
                "message": str(e),
                "type": "server_error",
            },
        }


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


def initialize_model():
    global encoder
    try:
        logger.info("Initializing VoiceEncoder model...")
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


def start_local_server(port: int = 8000):
    """Start a local FastAPI server that mimics RunPod's /runsync endpoint."""
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    import uvicorn
    import uuid

    app = FastAPI(title="Resemblyzer Voice API (local)")

    @app.post("/runsync")
    @app.post("/run")
    async def runsync(request: Request):
        body = await request.json()
        job_id = str(uuid.uuid4())[:8]
        try:
            output = handler({"input": body.get("input", body)})
            return JSONResponse({
                "id": job_id,
                "status": "COMPLETED",
                "output": output,
            })
        except Exception as e:
            return JSONResponse({
                "id": job_id,
                "status": "FAILED",
                "error": str(e),
            }, status_code=500)

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": MODEL_ID}

    logger.info(f"Starting local server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    if not initialize_model():
        logger.error("Failed to initialize model. Exiting.")
        exit(1)

    # RunPod production: use serverless handler
    # Local dev: start FastAPI server on port 8000
    if os.environ.get("RUNPOD_POD_ID") or os.environ.get("RUNPOD_API_KEY"):
        logger.info("Starting RunPod serverless handler...")
        runpod.serverless.start({"handler": handler})
    else:
        start_local_server(port=int(os.environ.get("PORT", "8000")))
