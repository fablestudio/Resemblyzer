# Resemblyzer Voice API

Stateless voice embedding and speaker identification service. Runs on RunPod serverless or locally via Docker.

**This container is a pure audio compute service** — it has no database access. The webapp (project-delta) owns all Supabase operations and passes pre-computed embeddings or audio URLs directly.

## Endpoints

All requests use RunPod's `/runsync` wrapper. The `endpoint` field in `input` routes to the correct handler.

---

### POST /v1/embeddings

Generate voice embeddings from audio. Returns raw 256-dimensional vectors.

```json
{
  "input": {
    "endpoint": "/v1/embeddings",
    "audio_url": "https://example.com/voice_sample.mp3"
  }
}
```

**Batch mode:**

```json
{
  "input": {
    "endpoint": "/v1/embeddings",
    "audio_files": [
      { "audio_url": "https://example.com/sample1.mp3" },
      { "audio_url": "https://example.com/sample2.mp3" }
    ]
  }
}
```

**Response:**

```json
{
  "object": "list",
  "data": [
    { "object": "embedding", "index": 0, "embedding": [0.0123, -0.0456, ...] }
  ],
  "model": "resemblyzer-v1",
  "usage": { "prompt_tokens": 0, "total_tokens": 0 }
}
```

---

### POST /v1/audio/identify

Speaker identification. Reference speakers are provided as pre-computed embeddings and/or audio URLs.

#### Reference speaker inputs

| Field | Description |
| --- | --- |
| `reference_embeddings` | `{voice_id: [256 floats]}` — pre-computed embeddings (from DB cache) |
| `reference_audio_urls` | `{voice_id: "url"}` — audio to embed on-the-fly (for cache misses) |

Embeddings take precedence. Any embeddings generated from `reference_audio_urls` are returned in `generated_embeddings` so the caller can cache them.

#### Segmentation modes

| `segmentation` value | Behavior |
| --- | --- |
| `"whole"` | Score entire audio as one segment |
| `"auto"` (default) | Sliding-window diarization |
| `{rate, resolution, ...}` | Auto with custom parameters |
| WebVTT string | Parse timestamps and score each cue |
| JSON array of `{start, end}` | Score each segment independently |

#### Parameters

| Param | Default | Description |
| --- | --- | --- |
| `audio_url` / `audio_base64` | — | Input audio (required) |
| `reference_embeddings` | `{}` | Pre-computed speaker embeddings |
| `reference_audio_urls` | `{}` | Audio URLs to embed on-the-fly |
| `top_k` | `5` | Max speakers per segment |
| `segmentation` | `"auto"` | Segmentation mode |
| `rate` | `4` | Partials per second (auto mode) |
| `resolution` | `0.5` | Bucket size in seconds (auto mode) |
| `threshold_confident` | `0.75` | Score threshold for "high" confidence |
| `threshold_uncertain` | `0.65` | Score threshold for "medium" confidence |

#### Example

```json
{
  "input": {
    "endpoint": "/v1/audio/identify",
    "audio_url": "https://example.com/conversation.mp3",
    "reference_embeddings": {
      "voice_alice": [0.08, 0.0, 0.006, ...]
    },
    "reference_audio_urls": {
      "voice_bob": "https://example.com/bob_sample.mp3"
    },
    "segmentation": "auto",
    "top_k": 3
  }
}
```

#### Response

```json
{
  "object": "speaker_identification",
  "model": "resemblyzer-v1",
  "duration": 51.84,
  "num_segments": 12,
  "segments": [
    {
      "start": 4.0,
      "end": 6.0,
      "duration": 2.0,
      "speaker": "voice_alice",
      "confidence": "high",
      "top_speakers": [
        { "voice_id": "voice_alice", "score": 0.8427 },
        { "voice_id": "voice_bob", "score": 0.5472 }
      ],
      "scores": { "voice_alice": 0.8427, "voice_bob": 0.5472 }
    }
  ],
  "speaker_summary": {
    "voice_alice": { "min": 0.37, "max": 0.93, "mean": 0.59 }
  },
  "generated_embeddings": {
    "voice_bob": [0.05, 0.0, 0.12, ...]
  }
}
```

---

## Architecture

```
Webapp (project-delta)              Resemblyzer Container
  │                                   │
  ├─ GET voice_embeddings from DB     │
  ├─ GET voices.preview_url           │
  │                                   │
  ├──── POST /runsync ───────────────►│
  │     reference_embeddings (cached)  │  ← pure audio compute
  │     reference_audio_urls (misses)  │  ← no DB access
  │                                   │
  │◄──── response ────────────────────│
  │     segments, scores              │
  │     generated_embeddings          │  ← new embeddings to cache
  │                                   │
  ├─ UPSERT generated_embeddings     │
  └─ return result to client          │
```

## Webapp API routes (project-delta)

| Route | Description |
| --- | --- |
| `POST /api/v1/voice/embeddings` | Generate & store voice embeddings |
| `GET /api/v1/voice/embeddings?voice_ids=a,b` | Retrieve stored embeddings |
| `DELETE /api/v1/voice/embeddings` | Delete a stored embedding |
| `POST /api/v1/voice/identify` | Speaker identification (orchestrates DB + Resemblyzer) |

## Local development

```bash
# Start Resemblyzer container
cd Resemblyzer
docker compose -f docker-compose.runpod.yml up -d

# Container serves on http://localhost:8000
# Webapp calls it directly — no Supabase credentials needed in container
```

## Audio input formats

- `*_url` — HTTP URL or local file path
- `*_base64` — raw audio bytes as base64

Supported: `.wav`, `.mp3`, `.m4a`, `.flac`

## Cold start

~2 minutes on RunPod (worker spin-up). Warm requests: 5-20 seconds.
