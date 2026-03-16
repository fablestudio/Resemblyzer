# Resemblyzer RunPod API

Endpoint ID: `v8dcvggw1jxkgu`

Base URL: `https://api.runpod.ai/v2/v8dcvggw1jxkgu`

## Authentication

All requests require a RunPod API key in the `Authorization` header:

```
Authorization: Bearer <RUNPOD_API_KEY>
```

## Modes

### 1. Identify (with automatic diarization)

Default when no `segments` are provided. Uses sliding-window embeddings to detect who is speaking when across the entire audio.

```json
{
  "input": {
    "mode": "identify",
    "input_audio_url": "https://example.com/conversation.mp3",
    "speakers": {
      "Alice": {
        "samples": [{ "audio_url": "https://example.com/alice_sample.mp3" }]
      },
      "Bob": {
        "samples": [{ "audio_url": "https://example.com/bob_sample.mp3" }]
      }
    }
  }
}
```

Optional diarization parameters:

| Param                 | Default | Description                                                       |
| --------------------- | ------- | ----------------------------------------------------------------- |
| `rate`                | `4`     | Partial utterances per second (higher = finer resolution, slower) |
| `resolution`          | `0.5`   | Bucket size in seconds for speaker assignment                     |
| `threshold_confident` | `0.75`  | Minimum score for "confident" label                               |
| `threshold_uncertain` | `0.65`  | Minimum score for "uncertain" label (below = "none")              |

Response:

```json
{
  "status": "success",
  "mode": "diarize",
  "duration": 51.84,
  "num_segments": 12,
  "segments": [
    {
      "start": 4.0,
      "end": 6.0,
      "duration": 2.0,
      "speaker": "Alice",
      "score": 0.8427,
      "confidence": "confident",
      "scores": { "Alice": 0.8427, "Bob": 0.5472 }
    }
  ],
  "speaker_summary": {
    "Alice": { "min": 0.37, "max": 0.93, "mean": 0.59 },
    "Bob": { "min": 0.37, "max": 0.97, "mean": 0.55 }
  }
}
```

### 2. Identify (with WebVTT segments)

When `segments` is provided, each subtitle cue is sliced from the audio and scored independently. Useful for verifying speaker labels in an existing transcript.

```json
{
  "input": {
    "mode": "identify",
    "input_audio_url": "https://example.com/conversation.mp3",
    "speakers": {
      "Alice": {
        "samples": [{ "audio_url": "https://example.com/alice_sample.mp3" }]
      },
      "Bob": {
        "samples": [{ "audio_url": "https://example.com/bob_sample.mp3" }]
      }
    },
    "segments": "WEBVTT\n\n1\n00:00:01.019 --> 00:00:02.899\n<v Alice>Hey, how are you?\n\n2\n00:00:03.439 --> 00:00:05.419\n<v Bob>I'm good, thanks!\n"
  }
}
```

Segments can also be passed as a JSON array:

```json
"segments": [
  { "start": 1.019, "end": 2.899, "speaker": "Alice", "text": "Hey, how are you?" },
  { "start": 3.439, "end": 5.419, "speaker": "Bob", "text": "I'm good, thanks!" }
]
```

Response:

```json
{
  "status": "success",
  "mode": "segmented",
  "num_segments": 2,
  "segments": [
    {
      "index": 1,
      "start": 1.019,
      "end": 2.899,
      "text": "Hey, how are you?",
      "speaker": "Alice",
      "scores": { "Alice": 0.82, "Bob": 0.55 },
      "best_match": "Alice",
      "best_score": 0.82
    }
  ]
}
```

### 3. Compare

Direct similarity between two audio files.

```json
{
  "input": {
    "mode": "compare",
    "audio_a_url": "https://example.com/clip1.mp3",
    "audio_b_url": "https://example.com/clip2.mp3"
  }
}
```

Response: `{ "status": "success", "similarity": 0.7857 }`

### 4. Embed

Generate raw 256-dimensional voice embeddings.

```json
{
  "input": {
    "mode": "embed",
    "audio_url": "https://example.com/clip.mp3"
  }
}
```

For multiple files:

```json
{
  "input": {
    "mode": "embed",
    "audio_files": [
      { "audio_url": "https://example.com/clip1.mp3" },
      { "audio_url": "https://example.com/clip2.mp3" }
    ]
  }
}
```

## Audio input formats

All audio parameters accept either a URL or base64-encoded data:

- `*_url` — the endpoint downloads and caches the file
- `*_base64` — raw audio bytes encoded as base64

Supported formats: `.wav`, `.mp3`, `.m4a`, `.flac`

## Next.js API Route (Vercel)

### Environment variable

Add to `.env.local` (and Vercel project settings):

```
RUNPOD_API_KEY=rpa_xxxxx
```

### Shared client — `lib/runpod.ts`

```ts
const RUNPOD_API_KEY = process.env.RUNPOD_API_KEY!;
const ENDPOINT_ID = "v8dcvggw1jxkgu";
const BASE_URL = `https://api.runpod.ai/v2/${ENDPOINT_ID}`;

export interface RunPodResponse<T = unknown> {
  id: string;
  status: "IN_QUEUE" | "IN_PROGRESS" | "COMPLETED" | "FAILED" | "TIMED_OUT";
  output?: T;
  delayTime?: number;
  executionTime?: number;
}

export async function runpodRequest<T = unknown>(
  input: Record<string, unknown>,
): Promise<RunPodResponse<T>> {
  // Try synchronous first (30s timeout on RunPod side)
  const res = await fetch(`${BASE_URL}/runsync`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${RUNPOD_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ input }),
  });

  if (!res.ok) {
    throw new Error(`RunPod request failed: ${res.status} ${await res.text()}`);
  }

  const data: RunPodResponse<T> = await res.json();

  // If still processing, poll for result
  if (data.status === "IN_QUEUE" || data.status === "IN_PROGRESS") {
    return pollForResult<T>(data.id);
  }

  return data;
}

async function pollForResult<T>(
  jobId: string,
  maxAttempts = 60,
  intervalMs = 2000,
): Promise<RunPodResponse<T>> {
  for (let i = 0; i < maxAttempts; i++) {
    await new Promise((r) => setTimeout(r, intervalMs));

    const res = await fetch(`${BASE_URL}/status/${jobId}`, {
      headers: { Authorization: `Bearer ${RUNPOD_API_KEY}` },
    });
    const data: RunPodResponse<T> = await res.json();

    if (
      data.status === "COMPLETED" ||
      data.status === "FAILED" ||
      data.status === "TIMED_OUT"
    ) {
      return data;
    }
  }

  throw new Error(`RunPod job ${jobId} timed out after polling`);
}
```

### Diarize endpoint — `app/api/resemblyzer/diarize/route.ts`

Automatic speaker diarization (no segments needed).

```ts
import { NextRequest, NextResponse } from "next/server";
import { runpodRequest } from "@/lib/runpod";

interface DiarizeSegment {
  start: number;
  end: number;
  duration: number;
  speaker: string | null;
  score: number;
  confidence: "confident" | "uncertain" | "none";
  scores: Record<string, number>;
}

interface DiarizeOutput {
  status: string;
  mode: string;
  duration: number;
  num_segments: number;
  segments: DiarizeSegment[];
  speaker_summary: Record<string, { min: number; max: number; mean: number }>;
}

export async function POST(req: NextRequest) {
  const body = await req.json();
  const { inputAudioUrl, speakers } = body;

  // speakers: { "Alice": { sampleUrls: ["https://..."] }, ... }
  const speakersPayload: Record<string, { samples: { audio_url: string }[] }> =
    {};
  for (const [name, data] of Object.entries(speakers) as [
    string,
    { sampleUrls: string[] },
  ][]) {
    speakersPayload[name] = {
      samples: data.sampleUrls.map((url) => ({ audio_url: url })),
    };
  }

  const result = await runpodRequest<DiarizeOutput>({
    mode: "identify",
    input_audio_url: inputAudioUrl,
    speakers: speakersPayload,
  });

  if (result.status === "FAILED") {
    return NextResponse.json({ error: "Diarization failed" }, { status: 500 });
  }

  return NextResponse.json(result.output);
}
```

### Segment scoring endpoint — `app/api/resemblyzer/score-segments/route.ts`

Score an existing WebVTT transcript against speaker samples.

```ts
import { NextRequest, NextResponse } from "next/server";
import { runpodRequest } from "@/lib/runpod";

interface SegmentResult {
  index: number;
  start: number;
  end: number;
  text: string;
  speaker: string | null;
  scores: Record<string, number>;
  best_match: string;
  best_score: number;
}

interface SegmentedOutput {
  status: string;
  mode: string;
  num_segments: number;
  segments: SegmentResult[];
}

export async function POST(req: NextRequest) {
  const body = await req.json();
  const { inputAudioUrl, speakers, webvtt } = body;

  const speakersPayload: Record<string, { samples: { audio_url: string }[] }> =
    {};
  for (const [name, data] of Object.entries(speakers) as [
    string,
    { sampleUrls: string[] },
  ][]) {
    speakersPayload[name] = {
      samples: data.sampleUrls.map((url) => ({ audio_url: url })),
    };
  }

  const result = await runpodRequest<SegmentedOutput>({
    mode: "identify",
    input_audio_url: inputAudioUrl,
    speakers: speakersPayload,
    segments: webvtt,
  });

  if (result.status === "FAILED") {
    return NextResponse.json(
      { error: "Segment scoring failed" },
      { status: 500 },
    );
  }

  return NextResponse.json(result.output);
}
```

### Compare endpoint — `app/api/resemblyzer/compare/route.ts`

```ts
import { NextRequest, NextResponse } from "next/server";
import { runpodRequest } from "@/lib/runpod";

export async function POST(req: NextRequest) {
  const { audioUrlA, audioUrlB } = await req.json();

  const result = await runpodRequest<{ status: string; similarity: number }>({
    mode: "compare",
    audio_a_url: audioUrlA,
    audio_b_url: audioUrlB,
  });

  if (result.status === "FAILED") {
    return NextResponse.json({ error: "Comparison failed" }, { status: 500 });
  }

  return NextResponse.json(result.output);
}
```

### Client-side usage example

```ts
// Diarize a conversation
const res = await fetch("/api/resemblyzer/diarize", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    inputAudioUrl: "https://storage.example.com/conversation.mp3",
    speakers: {
      "Ross Geller": { sampleUrls: ["https://storage.example.com/ross.mp3"] },
      "Rachel Green": {
        sampleUrls: ["https://storage.example.com/rachel.mp3"],
      },
    },
  }),
});
const data = await res.json();
// data.segments => [{ start: 4.0, end: 6.0, speaker: "Rachel Green", score: 0.84, ... }]

// Score a WebVTT transcript
const res2 = await fetch("/api/resemblyzer/score-segments", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    inputAudioUrl: "https://storage.example.com/conversation.mp3",
    speakers: {
      "Ross Geller": { sampleUrls: ["https://storage.example.com/ross.mp3"] },
      "Rachel Green": {
        sampleUrls: ["https://storage.example.com/rachel.mp3"],
      },
    },
    webvtt: `WEBVTT\n\n1\n00:00:01.019 --> 00:00:02.899\n<v Rachel Green>Hey!\n`,
  }),
});
```

## Cold start

First request after idle may take ~2 minutes (worker spin-up + model load). Subsequent requests complete in 5-20 seconds depending on audio length. The model loads in ~0.02s once the container is running.
