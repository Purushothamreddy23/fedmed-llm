-- Run this in your Supabase project → SQL Editor → New Query
-- Creates the two tables needed by FedMed-LLM backend

-- ── Users table ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email         TEXT UNIQUE NOT NULL,
    name          TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

-- ── Chat history table ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS chat_history (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id    UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    question   TEXT NOT NULL,
    answer     TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for fast history lookups per user
CREATE INDEX IF NOT EXISTS idx_chat_history_user_id
    ON chat_history(user_id, created_at DESC);

-- Row-level security (optional but good practice)
ALTER TABLE users        ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_history ENABLE ROW LEVEL SECURITY;
