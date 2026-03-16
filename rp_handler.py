import runpod
import logging
import hashlib
import os
import re
import base64
import tempfile
import requests
import numpy as np
from pathlib import Path

from resemblyzer import VoiceEncoder, preprocess_wav
from resemblyzer.hparams import sampling_rate

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


def parse_webvtt_timestamp(ts: str) -> float:
    """Parse a WebVTT timestamp (HH:MM:SS.mmm) to seconds."""
    parts = ts.strip().split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return float(parts[0])


def parse_webvtt(webvtt: str) -> list:
    """
    Parse WebVTT content into a list of segment dicts.

    Supports optional <v SpeakerName> voice tags in cue text.

    Returns list of:
      {"index": int, "start": float, "end": float, "text": str, "speaker": str|None}
    """
    segments = []
    # Split into cue blocks (separated by blank lines)
    blocks = re.split(r"\n\s*\n", webvtt.strip())

    for block in blocks:
        lines = block.strip().split("\n")
        if not lines:
            continue

        # Skip the WEBVTT header
        if lines[0].startswith("WEBVTT"):
            continue

        # Find the timestamp line
        ts_line = None
        ts_idx = None
        for i, line in enumerate(lines):
            if "-->" in line:
                ts_line = line
                ts_idx = i
                break

        if ts_line is None:
            continue

        # Parse timestamps
        match = re.match(r"([\d:.]+)\s*-->\s*([\d:.]+)", ts_line)
        if not match:
            continue
        start = parse_webvtt_timestamp(match.group(1))
        end = parse_webvtt_timestamp(match.group(2))

        # Parse cue index (line before timestamp, if numeric)
        cue_index = None
        if ts_idx > 0 and lines[ts_idx - 1].strip().isdigit():
            cue_index = int(lines[ts_idx - 1].strip())

        # Remaining lines are the cue text
        text_lines = lines[ts_idx + 1:]
        text = " ".join(text_lines).strip()

        # Extract <v SpeakerName> voice tag if present
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
    """Slice a waveform array by start/end seconds."""
    start_sample = int(start * sampling_rate)
    end_sample = int(end * sampling_rate)
    start_sample = max(0, start_sample)
    end_sample = min(len(wav), end_sample)
    return wav[start_sample:end_sample]


def handler(event):
    """
    RunPod serverless handler for Resemblyzer speaker identification.

    Supports three modes:

    1. "identify" - Compare input audio against named speaker samples.

       Without "segments": automatic continuous diarization using sliding-window
       embeddings (default). Detects who is speaking when by computing partial
       embeddings across the audio and merging adjacent windows with the same
       best-matching speaker. Optional params: rate (default 4), resolution
       (default 0.5), threshold_confident (default 0.75), threshold_uncertain
       (default 0.65).

       With "segments" (WebVTT string or list of {start, end} dicts): scores
       each segment independently against all speakers.

       Input:
       {
         "mode": "identify",
         "input_audio_url": "https://...",        # OR "input_audio_base64": "<base64>"
         "speakers": {
           "speaker_name": {
             "samples": [
               {"audio_url": "https://..."},      # OR {"audio_base64": "<base64>"}
             ]
           }
         },
         // Optional - omit for auto-diarization:
         "segments": "WEBVTT\n\n1\n00:00:01.019 --> 00:00:02.899\n<v Rachel>Hey!\n\n..."
         // OR "segments": [{"start": 1.019, "end": 2.899, "speaker": "Rachel", "text": "Hey!"}]
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
    """
    Compare input audio against named speaker samples.

    Three sub-modes determined by the presence of "segments":
      - segments provided (WebVTT or list): score each cue segment independently
      - no segments: automatic continuous diarization using sliding-window embeddings
    """
    global encoder

    # Resolve input audio
    input_path = resolve_audio(input_data, "input_audio")
    input_wav = preprocess_wav(input_path)

    # Build speaker embeddings
    speakers = input_data.get("speakers", {})
    if not speakers:
        return {"status": "error", "message": "No speakers provided."}

    speaker_embeds = {}
    for speaker_name, speaker_data in speakers.items():
        samples = speaker_data.get("samples", [])
        if not samples:
            logger.warning(f"No samples for speaker: {speaker_name}")
            continue
        sample_wavs = []
        for sample in samples:
            sample_path = resolve_audio(sample, "audio")
            sample_wav = preprocess_wav(sample_path)
            sample_wavs.append(sample_wav)
        speaker_embeds[speaker_name] = encoder.embed_speaker(sample_wavs)

    # Parse segments if provided
    raw_segments = input_data.get("segments")
    segments = None
    if raw_segments:
        if isinstance(raw_segments, str):
            segments = parse_webvtt(raw_segments)
        elif isinstance(raw_segments, list):
            segments = raw_segments

    if segments:
        return _identify_segmented(input_wav, speaker_embeds, segments)
    else:
        return _identify_diarize(input_wav, speaker_embeds, input_data)


def _identify_segmented(input_wav, speaker_embeds, segments) -> dict:
    """Score each provided segment (WebVTT cue or manual) against speakers."""
    global encoder

    logger.info(f"Processing {len(segments)} segments")
    segment_results = []

    for seg in segments:
        start = float(seg["start"])
        end = float(seg["end"])
        seg_wav = slice_wav_segment(input_wav, start, end)

        # Skip segments too short to embed (< 0.5s)
        min_samples = int(0.5 * sampling_rate)
        if len(seg_wav) < min_samples:
            logger.warning(f"Segment {seg.get('index', '?')} too short ({len(seg_wav)} samples), skipping")
            segment_results.append({
                "index": seg.get("index"),
                "start": start,
                "end": end,
                "text": seg.get("text", ""),
                "speaker": seg.get("speaker"),
                "scores": {name: 0.0 for name in speaker_embeds},
                "best_match": None,
                "best_score": 0.0,
                "skipped": True,
            })
            continue

        seg_embed = encoder.embed_utterance(seg_wav)

        scores = {}
        for name, spk_embed in speaker_embeds.items():
            scores[name] = round(float(np.dot(seg_embed, spk_embed)), 4)

        best_name = max(scores, key=scores.get)
        segment_results.append({
            "index": seg.get("index"),
            "start": start,
            "end": end,
            "text": seg.get("text", ""),
            "speaker": seg.get("speaker"),
            "scores": scores,
            "best_match": best_name,
            "best_score": scores[best_name],
        })

    return {
        "status": "success",
        "mode": "segmented",
        "num_segments": len(segment_results),
        "segments": segment_results,
    }


def _identify_diarize(input_wav, speaker_embeds, input_data) -> dict:
    """
    Automatic continuous diarization using sliding-window partial embeddings.

    Computes embeddings at a configurable rate across the entire audio,
    scores each window against all speakers, then merges adjacent windows
    with the same best speaker into contiguous segments.
    """
    global encoder

    # Configurable parameters
    rate = input_data.get("rate", 4)  # partial utterances per second
    threshold_confident = input_data.get("threshold_confident", 0.75)
    threshold_uncertain = input_data.get("threshold_uncertain", 0.65)
    resolution = input_data.get("resolution", 0.5)  # seconds per bucket

    duration = len(input_wav) / sampling_rate
    logger.info(f"Diarizing {duration:.1f}s of audio at rate={rate}")

    # Compute continuous partial embeddings across the full audio
    _, cont_embeds, wav_splits = encoder.embed_utterance(
        input_wav, return_partials=True, rate=rate
    )

    # Time midpoint for each partial embedding
    times = np.array([((s.start + s.stop) / 2) / sampling_rate for s in wav_splits])

    # Compute similarity matrix: each speaker vs all partial embeddings
    speaker_names = list(speaker_embeds.keys())
    similarity_dict = {
        name: cont_embeds @ embed for name, embed in speaker_embeds.items()
    }

    # Stack into matrix for argmax across speakers
    all_sims = np.array([similarity_dict[name] for name in speaker_names])

    # --- Build per-bucket speaker assignments at the given resolution ---
    total_dur = times[-1]
    buckets = []
    for t_start in np.arange(0, total_dur, resolution):
        t_end = t_start + resolution
        mask = (times >= t_start) & (times < t_end)
        if not mask.any():
            continue
        avg_sims = {name: float(similarity_dict[name][mask].mean()) for name in speaker_names}
        best = max(avg_sims, key=avg_sims.get)
        score = avg_sims[best]

        if score > threshold_confident:
            confidence = "confident"
        elif score > threshold_uncertain:
            confidence = "uncertain"
        else:
            confidence = "none"
            best = None

        buckets.append({
            "start": round(float(t_start), 3),
            "end": round(float(t_end), 3),
            "speaker": best,
            "score": round(score, 4),
            "confidence": confidence,
            "scores": {name: round(v, 4) for name, v in avg_sims.items()},
        })

    # --- Merge adjacent buckets with the same speaker into segments ---
    segments = []
    if buckets:
        current = {
            "start": buckets[0]["start"],
            "end": buckets[0]["end"],
            "speaker": buckets[0]["speaker"],
            "scores_sum": {name: buckets[0]["scores"][name] for name in speaker_names},
            "bucket_count": 1,
        }

        for b in buckets[1:]:
            if b["speaker"] == current["speaker"] and b["speaker"] is not None:
                current["end"] = b["end"]
                for name in speaker_names:
                    current["scores_sum"][name] += b["scores"][name]
                current["bucket_count"] += 1
            else:
                # Finalize current segment
                segments.append(_finalize_segment(
                    current, speaker_names, threshold_confident, threshold_uncertain
                ))
                current = {
                    "start": b["start"],
                    "end": b["end"],
                    "speaker": b["speaker"],
                    "scores_sum": {name: b["scores"][name] for name in speaker_names},
                    "bucket_count": 1,
                }

        # Finalize last segment
        segments.append(_finalize_segment(
            current, speaker_names, threshold_confident, threshold_uncertain
        ))

    # --- Per-speaker summary stats ---
    speaker_summary = {}
    for name in speaker_names:
        sims = similarity_dict[name]
        speaker_summary[name] = {
            "min": round(float(sims.min()), 4),
            "max": round(float(sims.max()), 4),
            "mean": round(float(sims.mean()), 4),
        }

    return {
        "status": "success",
        "mode": "diarize",
        "duration": round(duration, 3),
        "num_segments": len(segments),
        "segments": segments,
        "speaker_summary": speaker_summary,
    }


def _finalize_segment(current, speaker_names, threshold_confident, threshold_uncertain):
    """Convert a merged bucket group into a final segment dict."""
    n = current["bucket_count"]
    avg_scores = {name: round(current["scores_sum"][name] / n, 4) for name in speaker_names}
    best_score = avg_scores[current["speaker"]] if current["speaker"] else 0.0

    if best_score > threshold_confident:
        confidence = "confident"
    elif best_score > threshold_uncertain:
        confidence = "uncertain"
    else:
        confidence = "none"

    return {
        "start": current["start"],
        "end": current["end"],
        "duration": round(current["end"] - current["start"], 3),
        "speaker": current["speaker"],
        "score": best_score,
        "confidence": confidence,
        "scores": avg_scores,
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
