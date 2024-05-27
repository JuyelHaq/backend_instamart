# config.py

import os

class Config:
    DEBUG = False
    TESTING = False
    SECRET_KEY = 'your_secret_key'  # Change this to a random secret key
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    MAIL_SERVER = 'smtp.zoho.com'
    MAIL_PORT = 587
    MAIL_USE_TLS = True
    MAIL_USE_SSL = False
    MAIL_USERNAME = 'juyel@instamart.ai'
    MAIL_PASSWORD = 'Juyel257!'
    MAIL_DEFAULT_SENDER = 'juyel@instamart.ai'
    BASE_UPLOAD_FOLDER = '/home/clients'
    UPLOAD_FOLDER = BASE_UPLOAD_FOLDER
    UPLOAD_FOLDER_train = BASE_UPLOAD_FOLDER
    UPLOAD_FOLDER_db = BASE_UPLOAD_FOLDER
    CORS_SUPPORTS_CREDENTIALS = True
    CORS_ORIGIN = '*'
    CORS_EXPOSE_HEADERS = 'Authorization'
    
    DB_CONFIG = {
        'host': '127.0.0.1',
        'user': 'juyel',
        'password': '108@Xaplotes',
        'database': 'GenAIMaster',
    }

