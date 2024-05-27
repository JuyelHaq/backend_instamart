# config.py

import os

class Config:
    DEBUG = False
    TESTING = False
    SECRET_KEY = 'your_secret_key'  # Change this to a random secret key
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    MAIL_SERVER = 'smtp.gmail.com'
    MAIL_PORT = 587
    MAIL_USE_TLS = True
    MAIL_USE_SSL = False
    MAIL_USERNAME = 'juyel@thirdeyedata.ai'
    MAIL_PASSWORD = 'Iamjarav1#'
    MAIL_DEFAULT_SENDER = 'juyel@thirdeyedata.ai'
    CORS_SUPPORTS_CREDENTIALS = True
    CORS_ORIGIN = '*'
    CORS_EXPOSE_HEADERS = 'Authorization'
    
DB_CONFIG = {
        'host': '127.0.0.1',
        'user': 'juyel',
        'password': '108@Xaplotes',
        'database': 'GenAIMaster',
    }

