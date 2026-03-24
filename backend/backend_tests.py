"""
backend/backend_tests.py
FedMed-LLM Backend Test Suite (Day 7)

Run:
    cd backend
    pip install pytest httpx
    pytest backend_tests.py -v
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(__file__))

# ── Mock heavy dependencies before importing main ─────────────
# Mock supabase so tests don't need real DB
mock_supabase = MagicMock()
mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value.data = []
mock_supabase.table.return_value.insert.return_value.execute.return_value.data = [
    {"id": "test-user-id-123"}
]

# Mock model so tests don't need GPU
mock_model = MagicMock()
mock_tokenizer = MagicMock()
mock_tokenizer.eos_token_id = 0
mock_tokenizer.return_value = {"input_ids": MagicMock()}

with patch("supabase.create_client", return_value=mock_supabase), \
     patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model), \
     patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
     patch("os.path.exists", return_value=True):
    from main import app, hash_password, verify_password, create_jwt, check_rate_limit

client = TestClient(app)


# ── Auth Tests ────────────────────────────────────────────────
class TestAuth:

    def test_health_endpoint(self):
        """Health endpoint should return ok status."""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "uptime" in data
        assert "version" in data

    def test_password_hashing(self):
        """Password hash should verify correctly."""
        password = "test1234"
        hashed = hash_password(password)
        assert verify_password(password, hashed)
        assert not verify_password("wrongpassword", hashed)

    def test_jwt_creation(self):
        """JWT should be created and decoded correctly."""
        token = create_jwt("user-123")
        assert token is not None
        assert len(token) > 20

    def test_register_success(self):
        """Register with valid data should return token."""
        mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value.data = []
        mock_supabase.table.return_value.insert.return_value.execute.return_value.data = [
            {"id": "new-user-id"}
        ]
        response = client.post("/api/auth/register", json={
            "email": "newuser@test.com",
            "password": "password123",
            "name": "Test User"
        })
        assert response.status_code == 200
        assert "access_token" in response.json()

    def test_register_invalid_email(self):
        """Register with invalid email should fail."""
        response = client.post("/api/auth/register", json={
            "email": "notanemail",
            "password": "password123",
            "name": "Test User"
        })
        assert response.status_code == 422

    def test_register_short_password(self):
        """Register with short password should fail."""
        response = client.post("/api/auth/register", json={
            "email": "test@test.com",
            "password": "abc",
            "name": "Test User"
        })
        assert response.status_code == 422

    def test_login_wrong_password(self):
        """Login with wrong password should return 401."""
        from passlib.context import CryptContext
        pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")
        mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value.data = [{
            "id": "user-123",
            "email": "test@test.com",
            "password_hash": pwd.hash("correctpassword"),
        }]
        response = client.post("/api/auth/login", json={
            "email": "test@test.com",
            "password": "wrongpassword"
        })
        assert response.status_code == 401

    def test_login_user_not_found(self):
        """Login with non-existent user should return 401."""
        mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value.data = []
        response = client.post("/api/auth/login", json={
            "email": "nobody@test.com",
            "password": "password123"
        })
        assert response.status_code == 401


# ── Chat Validation Tests ─────────────────────────────────────
class TestChatValidation:

    def get_token(self):
        return create_jwt("test-user-id")

    def test_chat_requires_auth(self):
        """Chat without token should return 403."""
        response = client.post("/api/chat", json={"question": "test"})
        assert response.status_code == 403

    def test_chat_invalid_token(self):
        """Chat with invalid token should return 403."""
        response = client.post("/api/chat",
            json={"question": "test"},
            headers={"Authorization": "Bearer invalidtoken"}
        )
        assert response.status_code in [401, 403]

    def test_chat_question_too_long(self):
        """Question over 500 chars should return 422."""
        token = self.get_token()
        long_question = "a" * 501
        response = client.post("/api/chat",
            json={"question": long_question},
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 422

    def test_chat_empty_question(self):
        """Empty question should return 422."""
        token = self.get_token()
        response = client.post("/api/chat",
            json={"question": "   "},
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 422

    def test_history_requires_auth(self):
        """History without token should return 403."""
        response = client.get("/api/history")
        assert response.status_code == 403


# ── Rate Limiting Tests ───────────────────────────────────────
class TestRateLimiting:

    def test_rate_limit_allows_under_limit(self):
        """Should allow requests under the limit."""
        # Fresh user ID
        result = check_rate_limit("fresh-user-999")
        assert result is True

    def test_rate_limit_blocks_over_limit(self):
        """Should block after 10 requests in 60 seconds."""
        user_id = "rate-test-user-888"
        # Make 10 requests
        for _ in range(10):
            check_rate_limit(user_id)
        # 11th should be blocked
        result = check_rate_limit(user_id)
        assert result is False


# ── Input Validation Tests ────────────────────────────────────
class TestInputValidation:

    def test_question_max_length(self):
        """Question validator should reject > 500 chars."""
        from main import ChatRequest
        with pytest.raises(Exception):
            ChatRequest(question="x" * 501)

    def test_question_empty(self):
        """Question validator should reject empty string."""
        from main import ChatRequest
        with pytest.raises(Exception):
            ChatRequest(question="   ")

    def test_question_valid(self):
        """Valid question should pass."""
        from main import ChatRequest
        req = ChatRequest(question="What are symptoms of diabetes?")
        assert req.question == "What are symptoms of diabetes?"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
