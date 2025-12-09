import uuid
import jwt
import pytest
from rest_framework.test import APIClient

from monitoring.models import User
from monitoring import views


# Global DRF test client
client = APIClient()


def _make_jwt_for_user(user: User) -> str:
    """
    Create a JWT compatible with the real application settings.
    Uses the same secret and algorithm as in views._issue_jwt.
    """
    payload = {
        "sub": str(user.id),
        "username": user.username,
        "role": user.role,
    }
    return jwt.encode(payload, views.JWT_SECRET, algorithm=views.JWT_ALG)


@pytest.mark.django_db
def test_signin_returns_token_for_verified_user():
    """
    Real test for /auth/signin/:
    - Create a real User document (MongoEngine) with a UNIQUE username/email
    - Call /auth/signin/ with correct credentials
    - Expect 200 + token in the response
    """
    client.credentials()  # reset headers

    password = "pytest1234"
    suffix = uuid.uuid4().hex[:8]
    username = f"pytest-user-{suffix}"
    email = f"pytest-signin-{suffix}@example.com"

    user = User(
        username=username,
        full_name="Py Test",
        what_he_does="Testing",
        region="ariana",
        email=email,
        role="USER",
        is_active=True,
        is_email_verified=True,
    )
    user.set_password(password)
    user.save()

    response = client.post(
        "/auth/signin/",
        {"username": username, "password": password},
        format="json",
    )

    assert response.status_code == 200
    data = response.json()
    assert data["username"] == username
    assert "token" in data
    assert "token_expires" in data

    # Cleanup (optional, but nicer for your real DB)
    user.delete()


def test_aqi_without_token_returns_401():
    """
    Real call to /api/aqi/ WITHOUT Authorization header.
    Expect 401 from @require_jwt.
    """
    # Important: clear any previous Bearer token set by other tests
    client.credentials()

    response = client.get("/api/aqi/?region=ariana")

    assert response.status_code == 401
    assert response.json()["error"] == "Missing bearer token"


@pytest.mark.django_db
def test_aqi_with_real_ml_artifacts_returns_classification():
    """
    Full end-to-end AQI test:

    - Create a real User (unique username/email)
    - Build a real JWT with the same secret/alg as the app
    - Call /api/aqi/?region=ariana with Authorization header
    - This triggers classify_region_latest_window(region) using real ML artifacts
    - We verify the JSON structure of the response
    """
    client.credentials()  # reset headers

    suffix = uuid.uuid4().hex[:8]
    username = f"aqi-user-{suffix}"
    email = f"aqi_tester-{suffix}@example.com"

    user = User(
        username=username,
        full_name="AQI Tester",
        what_he_does="Testing AQI",
        region="ariana",
        email=email,
        role="USER",
        is_active=True,
        is_email_verified=True,
    )
    user.set_password("any-password")
    user.save()

    token = _make_jwt_for_user(user)
    client.credentials(HTTP_AUTHORIZATION=f"Bearer {token}")

    response = client.get("/api/aqi/?region=ariana")

    assert response.status_code == 200
    data = response.json()

    assert data["region"] == "ariana"
    assert "class_id" in data
    assert "class_name" in data
    assert "probabilities" in data
    assert isinstance(data["probabilities"], dict)

    # Cleanup
    user.delete()


@pytest.mark.django_db
def test_forecast_with_real_ml_artifacts_returns_forecast_and_aqi():
    """
    Full end-to-end forecast test:

    - Create a real User (unique username/email)
    - Build JWT
    - Call /api/forecast/?region=ariana&last_date=2023-12-31
    - This triggers forecast_region_next_day_dict + classify_forecast_for_region
      using the real ML code and artifacts
    - We verify the structure of the JSON response
    """
    client.credentials()  # reset headers

    suffix = uuid.uuid4().hex[:8]
    username = f"forecast-user-{suffix}"
    email = f"forecast_tester-{suffix}@example.com"

    user = User(
        username=username,
        full_name="Forecast Tester",
        what_he_does="Testing Forecast",
        region="ariana",
        email=email,
        role="USER",
        is_active=True,
        is_email_verified=True,
    )
    user.set_password("any-password")
    user.save()

    token = _make_jwt_for_user(user)
    client.credentials(HTTP_AUTHORIZATION=f"Bearer {token}")

    response = client.get(
        "/api/forecast/?region=ariana&last_date=2023-12-31"
    )

    assert response.status_code == 200
    data = response.json()

    # We don't assert exact numeric values, only structure & types
    assert "aqi_class" in data
    assert data["aqi_class"] in ("Good", "Moderate", "Unhealthy")
    assert isinstance(data.get("aqi_class_id"), int)
    assert "aqi_probabilities" in data
    assert isinstance(data["aqi_probabilities"], dict)

    # Cleanup
    user.delete()
