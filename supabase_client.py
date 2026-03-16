"""
Supabase client for voice embedding storage and retrieval.

Requires environment variables:
  SUPABASE_URL - Supabase project URL
  SUPABASE_SERVICE_KEY - Supabase service role key (bypasses RLS)
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_client = None


def get_client():
    """Lazy-initialize and return the Supabase client."""
    global _client
    if _client is not None:
        return _client

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables are required"
        )

    from supabase import create_client
    _client = create_client(url, key)
    logger.info("Supabase client initialized")
    return _client


def store_voice_embedding(voice_id: str, embedding: list[float]) -> dict:
    """Upsert a voice embedding into the voice_embeddings table."""
    client = get_client()
    embedding_str = f"[{','.join(str(x) for x in embedding)}]"
    result = (
        client.table("voice_embeddings")
        .upsert(
            {"voice_id": voice_id, "embedding": embedding_str},
            on_conflict="voice_id",
        )
        .execute()
    )
    logger.info(f"Stored embedding for voice_id={voice_id}")
    return result.data[0] if result.data else {}


def get_voice_embedding(voice_id: str) -> Optional[list[float]]:
    """Fetch a voice embedding by voice_id. Returns None if not found."""
    client = get_client()
    result = (
        client.table("voice_embeddings")
        .select("embedding")
        .eq("voice_id", voice_id)
        .limit(1)
        .execute()
    )
    if not result.data:
        return None
    raw = result.data[0]["embedding"]
    if isinstance(raw, str):
        raw = raw.strip("[]")
        return [float(x) for x in raw.split(",")]
    if isinstance(raw, list):
        return [float(x) for x in raw]
    return None


def get_voice_embeddings_batch(voice_ids: list[str]) -> dict[str, list[float]]:
    """Fetch embeddings for multiple voice_ids. Returns {voice_id: embedding}."""
    client = get_client()
    result = (
        client.table("voice_embeddings")
        .select("voice_id, embedding")
        .in_("voice_id", voice_ids)
        .execute()
    )
    embeddings = {}
    for row in result.data:
        raw = row["embedding"]
        if isinstance(raw, str):
            raw = raw.strip("[]")
            embeddings[row["voice_id"]] = [float(x) for x in raw.split(",")]
        elif isinstance(raw, list):
            embeddings[row["voice_id"]] = [float(x) for x in raw]
    return embeddings


def get_voice_preview_url(voice_id: str) -> Optional[str]:
    """Look up the preview_url for a voice_id from the voices table."""
    client = get_client()
    result = (
        client.table("voices")
        .select("preview_url")
        .eq("voice_id", voice_id)
        .eq("soft_deleted", False)
        .eq("deactivated", False)
        .limit(1)
        .execute()
    )
    if result.data and result.data[0].get("preview_url"):
        return result.data[0]["preview_url"]
    return None


def delete_voice_embedding(voice_id: str) -> bool:
    """Delete a voice embedding by voice_id."""
    client = get_client()
    result = (
        client.table("voice_embeddings")
        .delete()
        .eq("voice_id", voice_id)
        .execute()
    )
    return bool(result.data)
