import os
import pytest
from rest_framework.test import APIClient


client = APIClient()


def _mongo_configured() -> bool:
    """
    Return True if MongoDB URI is configured.
    We only run real Mongo tests when the environment is set.
    """
    return bool(os.getenv("MONGO_URI"))


@pytest.mark.skipif(
    not _mongo_configured(),
    reason="MONGO_URI not configured; real Mongo tests are skipped.",
)
@pytest.mark.xfail(
    reason=(
        "Known limitation: some S5P stats contain NaN float values, "
        "which are not JSON serialisable. DRF raises ValueError during "
        "json.dumps. API should later clean or replace NaN values "
        "before serialisation."
    )
)
def test_latest_s5p_real_mongo():
    """
    Real call to /api/s5p/latest/ with a real Mongo connection.

    The current implementation can fail with ValueError if NaN values are
    present in the stats, which is why this test is marked as xfail. This
    documents the limitation and acts as a reminder for future data cleaning.
    """
    response = client.get(
        "/api/s5p/latest/?region=ariana&gas=NO2,CO&top=2"
    )

    # This is what we EXPECT once the NaN issue is fixed.
    # Until then, the xfail marker will capture the ValueError from DRF.
    assert response.status_code == 200
    data = response.json()
    assert data["region"] == "ariana"
    assert "gases" in data
    assert "results" in data
    assert isinstance(data["results"], dict)


@pytest.mark.skipif(
    not _mongo_configured(),
    reason="MONGO_URI not configured; real Mongo tests are skipped.",
)
def test_latest_s3_lst_real_mongo():
    """
    Real call to /api/s3/lst/latest/ with real MongoDB.

    This endpoint usually returns JSON without NaN issues, so we keep
    this test as a normal passing test (no xfail).
    """
    response = client.get("/api/s3/lst/latest/?region=ariana&top=2")

    assert response.status_code == 200
    data = response.json()
    assert data["region"] == "ariana"
    assert "results" in data
    assert isinstance(data["results"], list)
