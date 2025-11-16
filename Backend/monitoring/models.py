import re
from datetime import datetime
from mongoengine import ( 
    Document, 
    StringField, 
    EmailField, 
    DateTimeField, 
    BooleanField,
    FloatField,
    DictField,
    )
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
    

class ContactMessage(Document):
    meta = {
        'collection': 'contact_messages',
        'db_alias': 'default',
    }
    name       = StringField(required=True, max_length=100)
    email      = EmailField(required=True)
    subject    = StringField(required=True, max_length=200)
    message    = StringField(required=True)
    created_at = DateTimeField(default=datetime.utcnow)
    is_read    = BooleanField(default=False)



class S5PDaily(Document):
    """
    Wrapper for the s5p_daily collection written by the S5P pipeline.

    We keep fields flexible with DictField so we don't fight with schema changes.
    """
    meta = {
        "collection": "s5p_daily",
        "db_alias": "default",
        "indexes": [
            {"fields": ["region", "gas", "date"], "name": "region_gas_date"},
            # helpful when you want latest by date
            {"fields": ["date"], "name": "by_date"},
        ],
    }

    # main query fields
    source       = StringField()          # "S5P"
    region       = StringField(required=True)
    gas          = StringField(required=True)      # "NO2", "CO", ...
    date         = StringField(required=True)      # "YYYY-MM-DD" (stored as string)
    grid_res_deg = FloatField()
    aoi_key      = StringField()

    # nested payload from the pipeline
    aoi          = DictField()            # GeoJSON polygon
    stats        = DictField()            # {mean, median, min, max, valid_pixels}
    files        = DictField()            # {tif, raw_nc}
    params       = DictField()            # bbox, start, end, qa_threshold, ...

    # stored as ISO string by the pipeline, so keep it simple
    created_at   = StringField()


class S3LSTDaily(Document):
    """
    Wrapper for the s3_lst_daily collection written by the S3 LST pipeline.
    """
    meta = {
        "collection": "s3_lst_daily",
        "db_alias": "default",
        "indexes": [
            {"fields": ["region", "date"], "name": "region_date"},
            {"fields": ["date"], "name": "by_date"},
        ],
    }

    source       = StringField()          # "S3"
    region       = StringField(required=True)
    product      = StringField()         # "LST"
    date         = StringField(required=True)      # "YYYY-MM-DD"
    grid_res_deg = FloatField()
    aoi_key      = StringField()

    aoi          = DictField()           # GeoJSON polygon
    stats        = DictField()           # {mean, median, min, max, valid_pixels}
    files        = DictField()           # {tif, raw_pkg}
    params       = DictField()           # bbox, start, end, ...

    created_at   = StringField()
