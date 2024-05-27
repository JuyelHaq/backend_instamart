from flask import Flask,session,Blueprint
from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO
import os, json
from flask import Flask, request, jsonify,send_file,g,session,Blueprint
from flask_cors import CORS
from flask_cors import cross_origin
import os
import stripe
from datetime import datetime
from werkzeug.utils import secure_filename
import pdfkit
from datetime import datetime



from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO
import os, json
from flask_mail import Mail, Message
from flask_cors import CORS


import uuid
from flask_mail import Mail, Message
import random
import string
import mysql.connector
from werkzeug.security import generate_password_hash, check_password_hash
from flask_jwt_extended import jwt_required, get_jwt_identity
from flask_jwt_extended import JWTManager
import secrets
import shutil
import os
from flask import Flask, jsonify, request,Blueprint
from PyPDF2 import PdfReader
import requests
from flask import g
from flask import request, redirect, flash,session,g
from werkzeug.utils import secure_filename
import os, shutil, validators
#from qa_over_docs import vector_db

import secrets
#from qa_over_docs import app
from flask import jsonify

UPLOAD_FOLDER ='uploads'
print(UPLOAD_FOLDER)
ALLOWED_EXTENSIONS = {'pdf', 'csv'}
CONTEXT_FILE = "context.json"
SOURCES_FILE = "sources.txt"


# Set up the Flask app with user-specific information


app = Flask(__name__)
#app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
#app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///project.db'
#app.config['DEFAULT_FOLDER_PATH'] = ''


app.config['SECRET_KEY'] = 'a super secret key'
#app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///project.db"
app.debug = True





socketio = SocketIO(app)
CORS(app)
#print("hhhh="+app.config['DEFAULT_FOLDER_PATH'])
#app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{app.config['DEFAULT_FOLDER_PATH']}/project.db"

#r_db = SQLAlchemy(app)

from qa_over_docs.apis.base import BaseAPI
from qa_over_docs.apis.openai import OpenAI
# from qa_over_docs.apis.huggingface import HuggingFace

api: BaseAPI = OpenAI()
db_config = {
        'host': '137.184.94.114',
        'user': 'juyel',
        'password': '108@Xaplotes',
        'database': 'GenAIMaster',
    }

BASE_UPLOAD_FOLDER='/home/clients/'
connection = mysql.connector.connect(**db_config)
cursor = connection.cursor()
print(cursor)


from qa_over_docs.views import auth


@app.before_request
def before_request():
    # Access the token from the Authorization header
    token = request.headers.get('Authorization')

    # Check if the token starts with 'Bearer ' and extract the actual token
    if token and token.startswith('Bearer '):
        actual_token = token.split('Bearer ')[1]
        g.api_token = actual_token
        print(g.api_token)
        get_user_id_query = 'SELECT user_id FROM Customers WHERE api_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()

        if user_id_result:
            user_id = user_id_result[0]

            # Retrieve the user's directory name from the Directory table
            get_directory_query = 'SELECT directory_name FROM Directory WHERE user_id = %s'
            cursor.execute(get_directory_query, (user_id,))
            directory_info = cursor.fetchone()
#
            if directory_info:
                directory_name = directory_info[0]
                user_directory_path = os.path.join(BASE_UPLOAD_FOLDER, directory_name)
                DEFAULT_FOLDER_PATH = user_directory_path
                app.config['DEFAULT_FOLDER_PATH']=DEFAULT_FOLDER_PATH
                print("defb"+DEFAULT_FOLDER_PATH)



        
    #else:
     #   api_token = None
     



#@app.before_request
#def before_request():
    # Access user_folder_name and user_folder_path from the request context
 #   user_folder_name = request.json.get('api_token')

    # Use user_folder_name and user_folder_path as needed
  #  if api_token:
   #     g.api_token = api_token
    #    print(g.api_token)

