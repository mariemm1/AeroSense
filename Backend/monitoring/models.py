import re
from datetime import datetime
from mongoengine import Document, StringField, EmailField, DateTimeField, BooleanField
from django.contrib.auth.hashers import make_password, check_password

# Removed ROLE_CHOICES – role is now free text

def slugify_username(full_name: str) -> str:
    s = full_name.strip().lower()
    s = re.sub(r'[^a-z0-9]+', '-', s)
    s = re.sub(r'-+', '-', s).strip('-')
    return s or 'user'

class User(Document):
    meta = {
        'collection': 'users',
        'db_alias': 'default',
        'indexes': [
            {'fields': ['username'], 'unique': True},
            {'fields': ['email'], 'unique': True},
        ],
    }

    # auth
    username      = StringField(required=True, unique=True, min_length=3, max_length=50)
    password_hash = StringField(required=True)

    # profile
    full_name     = StringField(required=True, max_length=120)
    what_he_does  = StringField(required=True, max_length=120)  # occupation/field
    region        = StringField(required=True, max_length=120)
    email         = EmailField(required=True, unique=True)

    # role is now a simple string (no choices) – default stays USER
    role          = StringField(default="USER", max_length=50)

    # housekeeping
    is_active     = BooleanField(default=True)
    created_at    = DateTimeField(default=datetime.utcnow)

    # email verification
    is_email_verified    = BooleanField(default=False)
    email_verify_token   = StringField()
    email_verify_expires = DateTimeField()

    def set_password(self, raw_password: str):
        self.password_hash = make_password(raw_password)

    def check_password(self, raw_password: str) -> bool:
        return check_password(raw_password, self.password_hash)
