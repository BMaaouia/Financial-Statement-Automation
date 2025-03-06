from pymongo import MongoClient
from django.conf import settings

def connect_to_mongodb():
    client = MongoClient(settings.MONGODB_URI)
    return client