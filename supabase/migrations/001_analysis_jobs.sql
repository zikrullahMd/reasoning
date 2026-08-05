-- Run in Supabase SQL Editor (Dashboard → SQL → New query)
-- Stores offline /analyze job metadata; PDFs remain in S3.

CREATE TABLE IF NOT EXISTS public.analysis_jobs (
    id TEXT PRIMARY KEY,
    job_type TEXT NOT NULL DEFAULT 'analyze',
    status TEXT NOT NULL DEFAULT 'queued',
    question TEXT NOT NULL,
    filename TEXT NOT NULL,
    input_storage TEXT NOT NULL,
    input_key TEXT NOT NULL,
    result_storage TEXT,
    result_key TEXT,
    result_text TEXT,
    callback_url TEXT,
    user_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    processed_at TIMESTAMPTZ,
    error TEXT,
    retry_count INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS analysis_jobs_status_created_idx
    ON public.analysis_jobs (status, created_at);

ALTER TABLE public.analysis_jobs ENABLE ROW LEVEL SECURITY;
