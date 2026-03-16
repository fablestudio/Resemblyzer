# Resemblyzer Voice API

OpenAI-compatible voice embedding and speaker identification API running on RunPod serverless.

Endpoint ID: `v8dcvggw1jxkgu`

Base URL: `https://api.runpod.ai/v2/v8dcvggw1jxkgu`

## Authentication

All requests require a RunPod API key:

```
Authorization: Bearer <RUNPOD_API_KEY>
```

## Environment Variables

The RunPod template requires:

| Variable | Description |
| --- | --- |
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_SERVICE_KEY` | Supabase service role key |

## Endpoints

All requests use RunPod's `/runsync` or `/run` wrapper. The `endpoint` field in `input` routes to the correct handler.

---

### POST /v1/embeddings

Generate voice embedding(s) and store in Supabase.

```json
{
  "input": {
    "endpoint": "/v1/embeddings",
    "voice_id": "abc123",
    "audio_url": "https://example.com/voice_sample.mp3"
  }
}
```

If `audio_url`/`audio_base64` is omitted, the handler looks up the `preview_url` from the `voices` table using the `voice_id`.

**Batch mode:**

```json
{
  "input": {
    "endpoint": "/v1/embeddings",
    "audio_files": [
      { "voice_id": "voice_1", "audio_url": "https://example.com/sample1.mp3" },
      { "voice_id": "voice_2" }
    ]
  }
}
```

**Response (OpenAI-compatible):**

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.0123, -0.0456, ...],
      "voice_id": "abc123"
    }
  ],
  "model": "resemblyzer-v1",
  "usage": { "prompt_tokens": 0, "total_tokens": 0 }
}
```

---

### POST /v1/embeddings/get

Retrieve a stored voice embedding by `voice_id`. If no embedding exists but the voice has a `preview_url` in the `voices` table, it auto-generates and stores the embedding.

```json
{
  "input": {
    "endpoint": "/v1/embeddings/get",
    "voice_id": "abc123"
  }
}
```

**Response:** Same OpenAI-compatible format as `/v1/embeddings`.

---

### POST /v1/embeddings/delete

Delete a stored voice embedding.

```json
{
  "input": {
    "endpoint": "/v1/embeddings/delete",
    "voice_id": "abc123"
  }
}
```

**Response:**

```json
{
  "deleted": true,
  "voice_id": "abc123",
  "object": "embedding.deleted"
}
```

---

### POST /v1/audio/identify

Speaker identification with per-segment confidence scores. Sends input audio and a list of `voice_ids` to compare against. Embeddings are automatically fetched from Supabase (or generated from `preview_url` if missing).

#### Segmentation modes

| `segmentation` value | Behavior |
| --- | --- |
| `"whole"` | Score entire audio as one segment |
| `"auto"` (default) | Sliding-window diarization with configurable `rate` and `resolution` |
| WebVTT string | Parse timestamps and score each cue |
| JSON array of `{start, end}` | Score each segment independently |

#### Parameters

| Param | Default | Description |
| --- | --- | --- |
| `audio_url` / `audio_base64` | — | Input audio to analyze (required) |
| `voice_ids` | — | List of voice_id strings (required) |
| `top_k` | `5` | Max speakers returned per segment |
| `segmentation` | `"auto"` | Segmentation mode (see above) |
| `rate` | `4` | Partial utterances per second (auto mode) |
| `resolution` | `0.5` | Bucket size in seconds (auto mode) |
| `threshold_confident` | `0.75` | Score threshold for "high" confidence |
| `threshold_uncertain` | `0.65` | Score threshold for "medium" confidence |

#### Example: Auto diarization

```json
{
  "input": {
    "endpoint": "/v1/audio/identify",
    "audio_url": "https://example.com/conversation.mp3",
    "voice_ids": ["voice_alice", "voice_bob"],
    "top_k": 3
  }
}
```

#### Example: Whole file

```json
{
  "input": {
    "endpoint": "/v1/audio/identify",
    "audio_url": "https://example.com/clip.mp3",
    "voice_ids": ["voice_alice", "voice_bob"],
    "segmentation": "whole"
  }
}
```

#### Example: WebVTT segments

```json
{
  "input": {
    "endpoint": "/v1/audio/identify",
    "audio_url": "https://example.com/conversation.mp3",
    "voice_ids": ["voice_alice", "voice_bob"],
    "segmentation": "WEBVTT\n\n1\n00:00:01.019 --> 00:00:02.899\n<v Alice>Hey!\n\n2\n00:00:03.439 --> 00:00:05.419\n<v Bob>Hi there!\n"
  }
}
```

#### Example: Manual segments

```json
{
  "input": {
    "endpoint": "/v1/audio/identify",
    "audio_url": "https://example.com/conversation.mp3",
    "voice_ids": ["voice_alice", "voice_bob"],
    "segmentation": [
      { "start": 1.019, "end": 2.899, "text": "Hey!" },
      { "start": 3.439, "end": 5.419, "text": "Hi there!" }
    ]
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
    "voice_alice": { "min": 0.37, "max": 0.93, "mean": 0.59 },
    "voice_bob": { "min": 0.37, "max": 0.97, "mean": 0.55 }
  }
}
```

Confidence levels:
- `"high"` — score > `threshold_confident` (default 0.75)
- `"medium"` — score > `threshold_uncertain` (default 0.65)
- `"low"` — score below both thresholds

---

## Error format

Errors follow the OpenAI error convention:

```json
{
  "error": {
    "message": "voice_id 'xyz' has no stored embedding and no preview_url.",
    "type": "invalid_request_error"
  }
}
```

## Audio input formats

All audio parameters accept either a URL or base64:

- `*_url` — downloads and caches the file
- `*_base64` — raw audio bytes encoded as base64

Supported formats: `.wav`, `.mp3`, `.m4a`, `.flac`

## Database setup

Run the migration in `migrations/001_create_voice_embeddings.sql` in your Supabase SQL editor. This creates:

- `voice_embeddings` table (voice_id, 256-dim vector embedding)
- `match_voice_embeddings()` function for similarity search
- RLS policy for service role access

The handler references the existing `voices` table (from `fable-simulation-webapp`) to look up `preview_url` for auto-generating embeddings.

## Cold start

First request after idle takes ~2 minutes (worker spin-up + model load). Subsequent requests complete in 5-20 seconds depending on audio length.
