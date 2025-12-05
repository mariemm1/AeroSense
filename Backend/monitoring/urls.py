from django.urls import path
from . import views

urlpatterns = [
    # Auth
    path("auth/signup/", views.signup, name="signup"),
    path("auth/signin/", views.signin, name="signin"),
    path("auth/verify-email/", views.verify_email, name="verify-email"),
    path("auth/resend-verification/", views.resend_verification, name="resend-verification"),

    # Contact
    path("api/contact/", views.contact, name="contact"),

    # Latest Copernicus stats
    path("api/s5p/latest/", views.latest_s5p, name="latest-s5p"),
    path("api/s3/lst/latest/", views.latest_s3_lst, name="latest-s3-lst"),

    # LSTM forecast (all gases + LST)
    path("api/forecast/", views.forecast_gases_next_day, name="forecast-gases-next-day"),

    # AQI classification (Good / Moderate / Unhealthy)
    path("api/aqi/", views.aqi_latest_classification, name="aqi-latest-classification"),
]
