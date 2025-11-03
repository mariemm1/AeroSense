# monitoring/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path("auth/signup/", views.signup, name="signup"),
    path("auth/signin/", views.signin, name="signin"),
    path("auth/verify-email/", views.verify_email, name="verify-email"),
    path("auth/resend-verification/", views.resend_verification, name="resend-verification"),
    
]
