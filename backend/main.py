"""
backend/main.py
FedMed-LLM FastAPI Backend

Endpoints:
  POST /api/auth/register   — Register new user
  POST /api/auth/login      — Login, returns JWT token
  POST /api/chat            — Ask a medical question (SSE streaming)
  GET  /api/history         — Get user's past conversations
  GET  /api/health          — Health check

Run locally:
    cd backend
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import os
import time
from datetime import datetime, timedelta
from typing import AsyncGenerator, Optional

import torch
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from passlib.context import CryptContext
from jose import JWTError, jwt
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
from supabase import create_client, Client

# ── Environment variables ─────────────────────────────────────
SUPABASE_URL    = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY    = os.getenv("SUPABASE_KEY", "")
JWT_SECRET      = os.getenv("JWT_SECRET", "change-this-in-production")
JWT_ALGORITHM   = "HS256"
JWT_EXPIRE_MINS = 60 * 24  # 24 hours

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "..", "model", "fedmed_phi2")
)

# ── App setup ─────────────────────────────────────────────────
app = FastAPI(
    title="FedMed-LLM API",
    description="Privacy-Preserving Medical Q&A via Federated LLM Fine-Tuning",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Restrict to your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Auth setup ────────────────────────────────────────────────
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer_scheme = HTTPBearer()

# ── Supabase ──────────────────────────────────────────────────
supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ── Global model (loaded at startup) ─────────────────────────
model_pipeline = None
tokenizer = None


@app.on_event("startup")
async def load_model():
    global model_pipeline, tokenizer

    if not os.path.exists(MODEL_PATH):
        print(f"WARNING: Model not found at {MODEL_PATH}")
        print("The /api/chat endpoint will return an error until the model is present.")
        return

    print(f"Loading FedMed-LLM model from {MODEL_PATH}...")
    start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model_pipeline = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model_pipeline.eval()

    elapsed = time.time() - start
    print(f"Model loaded in {elapsed:.1f}s")


# ── Schemas ───────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    email: str
    password: str
    name: str

class LoginRequest(BaseModel):
    email: str
    password: str

class ChatRequest(BaseModel):
    question: str
    max_tokens: int = 200

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


# ── Auth helpers ──────────────────────────────────────────────
def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def create_jwt(user_id: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=JWT_EXPIRE_MINS)
    return jwt.encode({"sub": user_id, "exp": expire}, JWT_SECRET, algorithm=JWT_ALGORITHM)

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)) -> str:
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user_id
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


# ── Endpoints ─────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model_pipeline is not None,
        "supabase_connected": supabase is not None,
    }


@app.post("/api/auth/register", response_model=TokenResponse)
async def register(body: RegisterRequest):
    if not supabase:
        raise HTTPException(status_code=503, detail="Database not configured")

    # Check if user exists
    existing = supabase.table("users").select("id").eq("email", body.email).execute()
    if existing.data:
        raise HTTPException(status_code=400, detail="Email already registered")

    # Create user
    hashed = hash_password(body.password)
    result = supabase.table("users").insert({
        "email": body.email,
        "name": body.name,
        "password_hash": hashed,
        "created_at": datetime.utcnow().isoformat(),
    }).execute()

    user_id = result.data[0]["id"]
    token = create_jwt(str(user_id))
    return TokenResponse(access_token=token)


@app.post("/api/auth/login", response_model=TokenResponse)
async def login(body: LoginRequest):
    if not supabase:
        raise HTTPException(status_code=503, detail="Database not configured")

    result = supabase.table("users").select("*").eq("email", body.email).execute()
    if not result.data:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    user = result.data[0]
    if not verify_password(body.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_jwt(str(user["id"]))
    return TokenResponse(access_token=token)


@app.post("/api/chat")
async def chat(
    body: ChatRequest,
    user_id: str = Depends(get_current_user),
):
    if model_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Ensure the federated training is complete."
        )

    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    prompt = f"Question: {question}\nAnswer:"

    async def token_stream() -> AsyncGenerator[str, None]:
        """Stream tokens via Server-Sent Events."""
        full_response = []

        inputs = tokenizer(prompt, return_tensors="pt").to(model_pipeline.device)
        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": body.max_tokens,
            "temperature": 0.7,
            "do_sample": True,
            "repetition_penalty": 1.1,
            "pad_token_id": tokenizer.eos_token_id,
        }

        # Run generation in a separate thread so we can stream
        thread = Thread(target=model_pipeline.generate, kwargs=generation_kwargs)
        thread.start()

        for token_text in streamer:
            full_response.append(token_text)
            yield f"data: {token_text}\n\n"

        yield "data: [DONE]\n\n"
        thread.join()

        # Save to Supabase after full response
        if supabase:
            try:
                supabase.table("chat_history").insert({
                    "user_id": user_id,
                    "question": question,
                    "answer": "".join(full_response),
                    "created_at": datetime.utcnow().isoformat(),
                }).execute()
            except Exception as e:
                print(f"Failed to save chat history: {e}")

    return StreamingResponse(
        token_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


@app.get("/api/history")
async def get_history(
    user_id: str = Depends(get_current_user),
    limit: int = 20,
):
    if not supabase:
        return {"history": []}

    result = supabase.table("chat_history") \
        .select("*") \
        .eq("user_id", user_id) \
        .order("created_at", desc=True) \
        .limit(limit) \
        .execute()

    return {"history": result.data or []}
