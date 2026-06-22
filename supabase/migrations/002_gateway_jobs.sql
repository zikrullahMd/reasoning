-- Unified offline job queue for all gateway endpoints.
-- Request bodies live in S3/local; metadata and results in this table.

CREATE TABLE IF NOT EXISTS public.gateway_jobs (
    id TEXT PRIMARY KEY,
    method TEXT NOT NULL,
    path TEXT NOT NULL,
    query_string TEXT NOT NULL DEFAULT '',
    route_target TEXT NOT NULL,
    headers_json TEXT NOT NULL DEFAULT '{}',
    content_type TEXT,
    body_storage TEXT NOT NULL,
    body_key TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    result_status_code INTEGER,
    result_content_type TEXT,
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

CREATE INDEX IF NOT EXISTS gateway_jobs_status_created_idx
    ON public.gateway_jobs (status, created_at);

ALTER TABLE public.gateway_jobs ENABLE ROW LEVEL SECURITY;
