-- Voice embeddings table for Resemblyzer (256-dim voice fingerprints)
-- Run this in Supabase SQL editor or add to your migration pipeline.
--
-- Requires pgvector extension (already enabled if you use character_embeddings).

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS public.voice_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    voice_id TEXT NOT NULL,
    embedding vector(256) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    CONSTRAINT voice_embeddings_voice_id_unique UNIQUE (voice_id)
);

CREATE INDEX IF NOT EXISTS idx_voice_embeddings_voice_id
    ON public.voice_embeddings (voice_id);

-- Similarity search function: find closest voice embeddings to a query
CREATE OR REPLACE FUNCTION match_voice_embeddings(
    query_embedding vector(256),
    match_threshold float DEFAULT 0.5,
    match_count int DEFAULT 10
)
RETURNS TABLE (
    voice_id TEXT,
    similarity float
)
LANGUAGE sql STABLE
AS $$
    SELECT
        ve.voice_id,
        1 - (ve.embedding <=> query_embedding) AS similarity
    FROM public.voice_embeddings ve
    WHERE 1 - (ve.embedding <=> query_embedding) > match_threshold
    ORDER BY similarity DESC
    LIMIT match_count;
$$;

-- RLS: allow service role full access (handler uses service key)
ALTER TABLE public.voice_embeddings ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Service role full access" ON public.voice_embeddings
    FOR ALL
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');
