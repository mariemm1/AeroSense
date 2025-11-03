# atmospheric_gases/mongo.py
"""
import os
from mongoengine import connect

MONGO_DB     = os.getenv("MONGO_DB", "monitoring")
MONGO_USER   = os.getenv("MONGO_USER", "monitoring")
MONGO_PASS   = os.getenv("MONGO_PASS", "monitoring")
MONGO_HOST   = os.getenv("MONGO_HOST", "mongodb+srv://mariemjabberi94:TestPwd1994@cluster0.eui54.mongodb.net/")

def init_mongo():
    # alias='default' lets MongoEngine use this connection implicitly
    connect(
        db=MONGO_DB,
        host=MONGO_HOST,
        username=MONGO_USER,
        password=MONGO_PASS,
        alias="default",
    )

"""
import os
from mongoengine import connect

MONGO_URI = os.getenv("MONGO_URI")

def init_mongo():
    if not MONGO_URI:
        raise RuntimeError("MONGO_URI not set")
    # Use alias "default" to match models' db_alias
    connect(host=MONGO_URI, alias="default")
