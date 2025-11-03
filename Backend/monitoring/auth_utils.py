import jwt
from functools import wraps
from django.conf import settings
from rest_framework.response import Response
from rest_framework import status

JWT_SECRET = getattr(settings, "SECRET_KEY", "dev-secret-change-me")
JWT_ALG = "HS256"

def require_jwt(view):
    @wraps(view)
    def _wrapped(request, *args, **kwargs):
        auth = request.META.get("HTTP_AUTHORIZATION", "")
        if not auth.startswith("Bearer "):
            return Response({"error": "Missing bearer token"}, status=status.HTTP_401_UNAUTHORIZED)
        token = auth.split(" ", 1)[1]
        try:
            request.jwt = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        except jwt.ExpiredSignatureError:
            return Response({"error": "Token expired"}, status=status.HTTP_401_UNAUTHORIZED)
        except jwt.InvalidTokenError:
            return Response({"error": "Invalid token"}, status=status.HTTP_401_UNAUTHORIZED)
        return view(request, *args, **kwargs)
    return _wrapped

def require_roles(*allowed):
    def decorator(view):
        @wraps(view)
        def _wrapped(request, *args, **kwargs):
            role = (getattr(request, "jwt", {}) or {}).get("role")
            if role not in allowed:
                return Response({"error": "Forbidden"}, status=status.HTTP_403_FORBIDDEN)
            return view(request, *args, **kwargs)
        return _wrapped
    return decorator
