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

from .models import User, slugify_username, ContactMessage

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



# monitoring/views.py (À LA FIN)

import logging
logger = logging.getLogger(__name__)

@api_view(["POST"])
@permission_classes([AllowAny])
def contact(request):
    data = request.data

    # --- Validation ---
    required_fields = ["name", "email", "subject", "message"]
    missing = [field for field in required_fields if not data.get(field)]
    if missing:
        return Response(
            {"error": f"Champs manquants : {', '.join(missing)}"},
            status=status.HTTP_400_BAD_REQUEST
        )

    name = data["name"].strip()
    email = data["email"].strip().lower()
    subject = data["subject"].strip()
    message = data["message"].strip()

    if len(name) < 2:
        return Response({"error": "Nom trop court."}, status=400)
    if len(subject) < 5:
        return Response({"error": "Sujet trop court."}, status=400)
    if len(message) < 10:
        return Response({"error": "Message trop court."}, status=400)
    if "@" not in email or "." not in email:
        return Response({"error": "Email invalide."}, status=400)

    # --- SAUVEGARDE MONGODB ---
    try:
        contact_msg = ContactMessage(
            name=name,
            email=email,
            subject=subject,
            message=message,
            is_read=False
        )
        contact_msg.save()
        logger.info(f"Message sauvegardé | ID: {contact_msg.id}")
    except Exception as e:
        logger.error(f"ÉCHEC SAUVEGARDE: {e}")
        return Response({"error": "Échec sauvegarde"}, status=500)

    # --- EMAIL À AEROSENSE ---
    try:
        logger.info(f"Envoi à contact@aerosense.com depuis {settings.DEFAULT_FROM_EMAIL}")
        send_mail(
            subject=f"[AeroSense] {subject}",
            message=f"""
            Nouveau message de contact AeroSense

            Nom: {name}
            Email: {email}
            Sujet: {subject}

            Message:
            {message}
            """,
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=['contact@aerosense.com'],
            fail_silently=False,
        )
        logger.info("Email AeroSense envoyé")
    except Exception as e:
        logger.error(f"ÉCHEC EMAIL AEROSENSE: {type(e).__name__}: {str(e)}")

    # --- EMAIL CONFIRMATION À L’UTILISATEUR ---
    try:
        logger.info(f"Envoi confirmation à {email}")
        send_mail(
            subject="Nous avons bien reçu votre message",
            message=f"""
            Bonjour {name},

            Merci pour votre message !

            Nous avons bien reçu votre demande :
            "{subject}"

            Nous vous répondrons sous 24h à l’adresse : {email}

            À bientôt,
            L’équipe AeroSense
            """,
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[email],
            fail_silently=False,
        )
        logger.info("Email confirmation envoyé")
    except Exception as e:
        logger.error(f"ÉCHEC EMAIL UTILISATEUR: {type(e).__name__}: {str(e)}")

    # SUCCÈS MÊME SI EMAIL ÉCHOUE
    return Response(
        {"detail": "Message envoyé avec succès !"},
        status=status.HTTP_201_CREATED
    )