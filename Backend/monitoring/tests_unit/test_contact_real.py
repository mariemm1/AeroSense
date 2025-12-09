import pytest
from django.test import override_settings
from rest_framework.test import APIClient

from monitoring.models import ContactMessage


client = APIClient()


@pytest.mark.django_db
@override_settings(EMAIL_BACKEND="django.core.mail.backends.locmem.EmailBackend")
def test_contact_creates_message_and_returns_201():
    """
    Real test of /api/contact/:
    - uses real view and MongoEngine model ContactMessage
    - uses locmem email backend (no real external emails sent)
    """
    initial_count = ContactMessage.objects.count()

    payload = {
        "name": "Test User",
        "email": "test.user@example.com",
        "subject": "Test subject",
        "message": "This is a long enough test message for AeroSense contact.",
    }

    response = client.post("/api/contact/", payload, format="json")

    assert response.status_code == 201
    data = response.json()
    assert "detail" in data

    # Check that a new ContactMessage has been persisted
    assert ContactMessage.objects.count() == initial_count + 1

    # We could also inspect django.core.mail.outbox here if needed.
