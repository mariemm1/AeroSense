from datetime import datetime, timedelta, timezone
from django.shortcuts import render
import os, jwt, secrets
from django.conf import settings
from django.urls import reverse
from django.core.mail import send_mail

from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework import status

from .models import User, slugify_username

# --- JWT config ---
JWT_SECRET = getattr(settings, "SECRET_KEY", "dev-secret-change-me")
JWT_ALG = "HS256"
JWT_EXPIRES_MIN = int(os.getenv("JWT_EXPIRES_MIN", "120"))

# --- Email verification config ---
EMAIL_VERIFY_TTL_MIN = int(os.getenv("EMAIL_VERIFY_TTL_MIN", "60"))  # token valid minutes


def _utcnow_naive():
    """Naive UTC (MongoEngine stores naive datetimes)."""
    return datetime.utcnow()

def _to_naive(dt):
    """Force any datetime to naive UTC for safe comparisons."""
    if dt is None:
        return None
    return dt.replace(tzinfo=None)

def _issue_jwt(user: User):
    now = datetime.now(timezone.utc)
    exp = now + timedelta(minutes=JWT_EXPIRES_MIN)
    payload = {
        "sub": str(user.id),
        "username": user.username,
        "role": user.role,
        "iat": int(now.timestamp()),
        "exp": int(exp.timestamp()),
        "iss": "atmospheric_gases",
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG), exp

def _unique_username_from_fullname(full_name: str) -> str:
    base = slugify_username(full_name)
    candidate, i = base, 1
    while User.objects(username=candidate).first():
        i += 1
        candidate = f"{base}-{i}"
    return candidate

def _build_verify_link(request, token: str) -> str:
    """
    Back-end verification link:
    http(s)://<host>/auth/verify-email/?token=<token>
    """
    path = reverse("verify-email")
    return request.build_absolute_uri(f"{path}?token={token}")

def _send_verification_email(request, user: User):
    """
    Generate single-use token with expiry and send a verification email.
    """
    token = secrets.token_urlsafe(32)
    user.email_verify_token = token
    user.email_verify_expires = _utcnow_naive() + timedelta(minutes=EMAIL_VERIFY_TTL_MIN)
    user.save()

    verify_url = _build_verify_link(request, token)

    subject = "Verify your email address"
    message = (
        f"Hi {user.full_name},\n\n"
        f"Please verify your email for your account '{user.username}'.\n\n"
        f"Click the link below (valid for {EMAIL_VERIFY_TTL_MIN} minutes):\n"
        f"{verify_url}\n\n"
        "If you did not request this, you can ignore this email."
    )

    send_mail(
        subject=subject,
        message=message,
        from_email=settings.DEFAULT_FROM_EMAIL,
        recipient_list=[user.email],
        fail_silently=False,
    )

@api_view(["POST"])
@permission_classes([AllowAny])
def signup(request):
    """
    Create account; send verification email.
    Role is free-form (e.g., RESEARCHER, REGULAR, STUDENT, TEACHER).
    No token returned here.
    """
    r = request.data
    required = ["full_name", "what_he_does", "region", "email", "password"]
    missing = [k for k in required if not r.get(k)]
    if missing:
        return Response({"error": f"Missing: {', '.join(missing)}"}, status=400)

    full_name    = r["full_name"].strip()
    what_he_does = r["what_he_does"].strip()
    region       = r["region"].strip()
    email        = r["email"].strip().lower()
    password     = r["password"]
    # Accept any role; default USER if not provided
    role         = (r.get("role") or "USER").strip()

    if len(password) < 6:
        return Response({"error": "Password must be at least 6 characters."}, status=400)
    if User.objects(email=email).first():
        return Response({"error": "Email already registered."}, status=409)

    username = _unique_username_from_fullname(full_name)
    user = User(
        username=username,
        full_name=full_name,
        what_he_does=what_he_does,
        region=region,
        email=email,
        role=role,
        is_active=True,
        is_email_verified=False,  # not verified yet
    )
    user.set_password(password)
    user.save()

    # Send verification email
    try:
        _send_verification_email(request, user)
    except Exception:
        return Response({"error": "Failed to send verification email."}, status=500)

    return Response({
        "id": str(user.id),
        "username": user.username,
        "email": user.email,
        "role": user.role,
        "message": "Account created. Please verify your email to sign in.",
        "verification_email_sent": True
    }, status=201)

@api_view(["POST"])
@permission_classes([AllowAny])
def signin(request):
    """
    Sign in by username; return JWT only if email is verified.
    """
    r = request.data
    username = (r.get("username") or "").strip().lower()
    password = r.get("password") or ""

    user = User.objects(username=username).first()
    if not user or not user.is_active or not user.check_password(password):
        return Response({"error": "Invalid credentials."}, status=401)

    if not getattr(user, "is_email_verified", False):
        return Response(
            {"error": "Email not verified. Please check your inbox.", "needs_verification": True},
            status=403
        )

    token, exp = _issue_jwt(user)
    return Response({
        "id": str(user.id),
        "username": user.username,
        "email": user.email,
        "role": user.role,
        "token": token,
        "token_expires": exp.isoformat()
    })

@api_view(["GET"])
@permission_classes([AllowAny])
def verify_email(request):
    """
    GET /auth/verify-email/?token=...
    Marks the user as verified if token is valid and not expired.
    """
    token = request.GET.get("token", "")
    if not token:
        return Response({"error": "Missing token."}, status=400)

    # naive UTC comparison to match MongoEngine storage
    now = _utcnow_naive()
    user = User.objects(email_verify_token=token).first()
    if not user:
        return Response({"error": "Invalid token."}, status=400)

    expires = _to_naive(user.email_verify_expires)
    if not expires or now > expires:
        return Response({"error": "Token expired."}, status=400)

    user.is_email_verified = True
    user.email_verify_token = None
    user.email_verify_expires = None
    user.save()

    return Response({"message": "Email verified successfully. You may now sign in."})

@api_view(["POST"])
@permission_classes([AllowAny])
def resend_verification(request):
    """
    POST { "username": "<username>" }  OR  { "email": "<email>" }
    Sends a new verification email if not yet verified.
    """
    login = (request.data.get("username") or request.data.get("email") or "").strip().lower()
    if not login:
        return Response({"error": "Provide username or email."}, status=400)

    user = User.objects(__raw__={"$or": [{"username": login}, {"email": login}]}).first()
    if not user:
        return Response({"error": "User not found."}, status=404)
    if getattr(user, "is_email_verified", False):
        return Response({"message": "Email already verified."})

    _send_verification_email(request, user)
    return Response({"message": "Verification email sent."})
