import jwt
from rest_framework.response import Response
from rest_framework.test import APIRequestFactory

from monitoring.auth_utils import require_jwt, JWT_SECRET, JWT_ALG

factory = APIRequestFactory()


@require_jwt
def _dummy_view(request):
    """A tiny real view protected by require_jwt."""
    return Response({"ok": True, "sub": request.jwt.get("sub")})


def _make_token(payload: dict) -> str:
    """Create a JWT with the same secret/alg as the app."""
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)


def test_require_jwt_missing_header_returns_401():
    """If no Authorization header → 401 with proper error."""
    request = factory.get("/dummy")
    response = _dummy_view(request)

    assert response.status_code == 401
    assert response.data["error"] == "Missing bearer token"


def test_require_jwt_invalid_token_returns_401():
    """If token is invalid → 401 invalid token."""
    request = factory.get(
        "/dummy", HTTP_AUTHORIZATION="Bearer this_is_not_valid"
    )
    response = _dummy_view(request)

    assert response.status_code == 401
    # one of the two messages depending on jwt version
    assert response.data["error"] in ("Invalid token", "Invalid token.")
    

def test_require_jwt_valid_token_calls_view_and_sets_request_jwt():
    """Valid token → real decorator lets the view run."""
    token = _make_token({"sub": "123", "role": "USER"})
    request = factory.get("/dummy", HTTP_AUTHORIZATION=f"Bearer {token}")

    response = _dummy_view(request)

    assert response.status_code == 200
    assert response.data["ok"] is True
    assert response.data["sub"] == "123"
