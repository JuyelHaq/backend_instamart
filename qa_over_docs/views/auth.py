from flask import Flask, request, jsonify,send_file,g,session,Blueprint
from flask_cors import CORS
from flask_cors import cross_origin
import os
import stripe
from datetime import datetime
from werkzeug.utils import secure_filename
import pdfkit
from datetime import datetime
from langchain.chat_models import ChatOpenAI


from langchain.chat_models import ChatOpenAI

from flask import Flask, request, jsonify,send_file
from flask_cors import CORS
from flask_cors import cross_origin
import os
import stripe
from datetime import datetime
from werkzeug.utils import secure_filename
import pdfkit
from datetime import datetime
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
from flask import Flask, jsonify, request
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma

from flask import Flask, request, jsonify
import os
import logging
import weasyprint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


#from convert import convert_to_md
from PyPDF2 import PdfReader
import requests
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from flask_cors import CORS

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
from qa_over_docs import app,context, r_db, ALLOWED_EXTENSIONS, UPLOAD_FOLDER, CONTEXT_FILE, SOURCES_FILE
import secrets
#from qa_over_docs import r_rb
from flask import jsonify
#auth_bp = Blueprint('auth', __name__)
from flask import request, redirect, flash
from werkzeug.utils import secure_filename
import os, shutil, validators
import subprocess

UPLOAD_FOLDER = 'uploads'

db_config = {
        'host': '137.184.94.114',
        'user': 'juyel',
        'password': '108@Xaplotes',
        'database': 'GenAIMaster',
    }
app.config['MAIL_SERVER'] = 'smtp.zoho.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_USERNAME'] = 'instamart.ai.support@instamart.ai'
app.config['MAIL_PASSWORD'] = 'Neoli257!'
app.config['MAIL_DEFAULT_SENDER'] = 'instamart.ai.support@instamart.ai'
from functools import wraps

mail=Mail(app)
# Set the Flask app's secret key
BASE_UPLOAD_FOLDER='/home/clients'
#connection = mysql.connector.connect(**db_config)
#cursor = connection.cursor()

#print(cursor)
def get_database_connection():
    return mysql.connector.connect(**db_config)


def send_error_email(api_name, error_message):
    try:
        subject = "API Error Notification"
        recipients = ["juyel@instamart.ai"]  # Add your email address here
        body = f"An error occurred in the {api_name} API:\n\n{error_message}"
        message = Message(subject=subject, recipients=recipients, body=body)
        mail.send(message)
    except Exception as e:
        logger.error(f"Error sending error email: {e}")



@app.route('/verify_token', methods=['POST'])

def verify_token():
    try:
        data = request.json
        token = data.get('api_token')

        if not token:
            return jsonify({'error': 'Missing token'}), 400

        # Establish database connection
        connection = get_database_connection()
        cursor = connection.cursor()

        check_token_query = 'SELECT user_id, first_name, last_name FROM Customers WHERE api_token = %s'
        cursor.execute(check_token_query, (token,))
        user_info = cursor.fetchone()

        if not user_info:
            cursor.close()
            connection.close()
            return jsonify({'error': 'Invalid token'}), 401

        user_id, first_name, last_name = user_info
        username = f"{first_name} {last_name}"
        session['user_id'] = user_id

        cursor.close()
        connection.close()

        user_id = session.get('user_id')
        print(user_id)

        return jsonify({'status': 'success', 'user_id': user_id, 'first_name': first_name, 'last_name': last_name}), 200

    except Exception as e:
        print(f"Error: {e}")
        send_error_email('verify_token', str(e))

        return jsonify({'error': 'An error occurred while processing the request'}), 500


@app.route('/setup_smtp', methods=['POST'])

def setup_smtp():
    connection = None
    cursor = None
    try:
        connection = get_database_connection()
        cursor = connection.cursor()

        # Retrieve token from headers
        token = request.headers.get('Authorization')
        if not token or not token.startswith('Bearer '):
            return jsonify({'error': 'Invalid token'}), 401

        actual_token = token.split('Bearer ')[1]

        # Retrieve user ID based on token
        get_user_id_query = 'SELECT user_id FROM Customers WHERE api_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()

        if not user_id_result:
            return jsonify({'error': 'User not found'}), 404

        user_id = user_id_result[0]

        # Retrieve SMTP details from request
        data = request.json
        print(data)
        port=data.get('port')
        smtp_server = data.get('smtp_server')
        email = data.get('email')
        password = data.get('password')

        # Upsert SMTP details for the user
        upsert_smtp_query = """
            INSERT INTO smtp_details (user_id,port, smtp_server, email, password)
            VALUES (%s, %s, %s, %s,%s)
            ON DUPLICATE KEY UPDATE smtp_server = VALUES(smtp_server), email = VALUES(email), password = VALUES(password),port=VALUES(port)
        """
        cursor.execute(upsert_smtp_query, (user_id, port, smtp_server, email, password))

        connection.commit()

        return jsonify({'message': 'SMTP server setup successful'}), 200

    except Exception as e:
        send_error_email('setup_smtp', str(e))
 
        return jsonify({'error': str(e)}), 500

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


@app.route('/register', methods=['POST'])
def register():
    print("Received a request to /register")

    data = request.json

    email = data.get('email')
    first_name = data.get('first_name')
    last_name = data.get('last_name')
    password = data.get('password')
    password_confirmation = data.get('password_confirmation')

    # Check if required fields are present
    if not email or not first_name or not last_name or not password or not password_confirmation:
        return jsonify({'error': 'Missing required fields'}), 400

    try:
        # Establish database connection
        connection = get_database_connection()
        cursor = connection.cursor()

        # Check if the password and confirmation match
        if password != password_confirmation:
            return jsonify({'error': 'Password and confirmation do not match'}), 400

        # Check if the email is already registered
        check_email_query = 'SELECT user_id FROM Customers WHERE email = %s'
        cursor.execute(check_email_query, (email,))
        existing_user = cursor.fetchone()
        if existing_user:
            return jsonify({'error': 'Email is already registered'}), 400

        # Hash the password before storing it
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

        # Insert the new user into the users table
        user_token = secrets.token_hex(16)
        insert_user_query = '''
        INSERT INTO Customers (email, first_name, last_name, password,user_token)
        VALUES (%s, %s, %s, %s,%s)
        '''
        cursor.execute(insert_user_query, (email, first_name, last_name, hashed_password,user_token))
        connection.commit()

        # Generate a unique api_token for the new user
        api_token = secrets.token_hex(16)

        # Update the user with the generated api_token
        update_token_query = 'UPDATE Customers SET api_token = %s WHERE email = %s'
        cursor.execute(update_token_query, (api_token, email))
        connection.commit()

        # Create a user folder using the generated token
        user_id_query = 'SELECT user_id FROM Customers WHERE email = %s'
        cursor.execute(user_id_query, (email,))
        user_id = cursor.fetchone()[0]

        user_folder_name = f"{first_name}_{user_id}"
        user_folder_path = os.path.join(BASE_UPLOAD_FOLDER, user_folder_name)
        subprocess.run(['sudo', 'useradd', '-m', '-d', user_folder_path, user_folder_name])
        subprocess.run(['sudo', 'chpasswd'], input=f"{user_folder_name}:{user_folder_name}", text=True)

        if not os.path.exists(user_folder_path):
            os.makedirs(user_folder_path)
        upload_folder_path = os.path.join(user_folder_path, UPLOAD_FOLDER)
        if not os.path.exists(upload_folder_path):
            os.makedirs(upload_folder_path)

        # Insert directory information into the database
        insert_directory_query = '''
        INSERT INTO Directory (user_id, directory_name)
        VALUES (%s, %s)
        '''
        cursor.execute(insert_directory_query, (user_id, user_folder_name))
        connection.commit()
        user_token_query = 'SELECT user_token FROM Customers WHERE user_id = %s'
        cursor.execute(user_token_query, (user_id,))
        usertoken = cursor.fetchone()[0]


        token_date = 'SELECT registration_date FROM Customers WHERE user_id = %s'
        cursor.execute(token_date, (user_id,))
        user_date = cursor.fetchone()

        if not user_date:
            return jsonify({'error': 'User not found'}), 404

        registration_date = user_date[0]

        registration_date = registration_date.date()
        current_date = datetime.now().date()
        days_registered = (current_date - registration_date).days
        if days_registered > 1:
    # Generate a new user token
           new_user_token = secrets.token_hex(16)

    # Update the user token in the database
           update_token_query = 'UPDATE Customers SET user_token = %s WHERE user_id = %s'
           cursor.execute(update_token_query, (new_user_token, user_id))
           connection.commit()


        # Send registration email
        send_email(email,api_token,usertoken)

        # Close cursor and connection
        cursor.close()
        connection.close()

        return jsonify({'api_token': api_token, 'message': 'Registration and login successful'}), 200

    except Exception as e:
        print(f"Error: {e}")
        send_error_email('register', str(e)) 
        return jsonify({'error': 'An error occurred while processing the request'}), 500


def send_email(recipient,verification_token,usertoken):
   # with app.app_context():
        verification_link = f"https://gentai.instamart.ai/verify?token={verification_token}"

        subject="Email Verification"
        msg = Message(subject, recipients=[recipient])
        msg.body = f"Please click the following link to verify your email: {verification_link} and here also your encrypted token which will be expired after 7 days \n\nClient Token: {usertoken}\nEmail: {recipient}"

        mail.send(msg)







import os
import shutil
from flask import Flask, jsonify

import fitz  # PyMuPDF
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma

def get_data_and_chroma_paths():
    connection = get_database_connection()
    cursor = connection.cursor()
    try:
        token = request.headers.get('Authorization')
        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
        else:
            actual_token = None
            response_data = {"message": "Invalid token format"}
            return None, None

        get_user_id_query = 'SELECT user_id FROM Customers WHERE api_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()

        if user_id_result:
            user_id = user_id_result[0]

            get_directory_query = 'SELECT directory_name FROM Directory WHERE user_id = %s'
            cursor.execute(get_directory_query, (user_id,))
            directory_info = cursor.fetchone()

            if directory_info:
                directory_name = directory_info[0]
                user_directory_path = os.path.join(BASE_UPLOAD_FOLDER, directory_name, UPLOAD_FOLDER)
                data_path = user_directory_path
                user_directory=os.path.join(BASE_UPLOAD_FOLDER, directory_name,"chroma")
                chroma_path = user_directory
                return data_path, chroma_path
    except Exception as e:
        send_error_email('generate_data_store', str(e))
        print("An error occurred:", e)
    finally:
        cursor.close()
        connection.close()
    return None, None
@app.route("/generate_data_store", methods=["POST"])
def generate_data_store():
    data_path, chroma_path = get_data_and_chroma_paths()
    if data_path and chroma_path:
        documents = load_documents(data_path)
        chunks = split_text(documents)
        save_to_chroma(chunks)
        
        return jsonify({"message": "Data store generated successfully!"})
    else:
        return jsonify({"error": "Failed to retrieve data and chroma paths."}), 500

def load_documents(data_path):
    documents = []
    for filename in os.listdir(data_path):
        if filename.endswith(".pdf"):
            full_path = os.path.join(data_path, filename)
            text = extract_text_from_pdf(full_path)
            document = Document(filename=filename, page_content=text)
            documents.append(document)
    return documents

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    for document in chunks:
        try:
            print(document.page_content)
        except UnicodeEncodeError:
            print("UnicodeEncodeError occurred, skipping printing for this document.")

        print(document.metadata)

    return chunks

def save_to_chroma(chunks: list[Document]):
    data_path, chroma_path = get_data_and_chroma_paths()
    print(chroma_path)
    os.makedirs(chroma_path, exist_ok=True)

    # Create a new DB from the documents.
    try:
        db = Chroma.from_documents(chunks, OpenAIEmbeddings(), persist_directory=chroma_path)
        db.persist()
        print(f"Saved {len(chunks)} chunks to {chroma_path}.")
    except Exception as e:
        send_error_email('generate_data_store', str(e))
        print(f"Error occurred while saving to Chroma: {e}")

import argparse
import sys
import os
from flask import Flask, request, jsonify
from dataclasses import dataclass
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
encoded_key = b'b0ZibHJtbTZ4ZXZKSmYyc1Z4ek5UYllkS01ZRmZGc2trLXBQNDVSMVQtND0='


def get_data_and_chroma_path_web():
    connection = get_database_connection()  # Assuming you have a function to establish a database connection
    cursor = connection.cursor()
    try:

       token = request.headers.get('Authorization')
       if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
       else:
            actual_token = None
            response_data = {"message": "Invalid token format"}
            return None, None

       get_user_id_query = 'SELECT user_id FROM Customers WHERE user_token = %s'
       cursor.execute(get_user_id_query, (actual_token,))
       user_id_result = cursor.fetchone()

       if user_id_result:
            user_id = user_id_result[0]

       get_directory_query = 'SELECT directory_name FROM Directory WHERE user_id = %s'
       cursor.execute(get_directory_query, (user_id,))
       directory_info = cursor.fetchone()

       if directory_info:
                directory_name = directory_info[0]
                user_directory_path = os.path.join(BASE_UPLOAD_FOLDER, directory_name, UPLOAD_FOLDER)
                data_path = user_directory_path
                user_directory = os.path.join(BASE_UPLOAD_FOLDER, directory_name,'chroma')
                chroma_path = user_directory
                print(chroma_path)
                return data_path, chroma_path
    except Exception as e:
        send_error_email('query_web', str(e))
        print("An error occurred:", e)
    finally:
        cursor.close()
        connection.close()
    return None, None
@app.route("/query_web", methods=["POST"])
def query_chroma_web():
    PROMPT_TEMPLATE = """


    # You will be acting as an AI PDF Expert named GentAI.
    # Your goal is to provide accurate answers and insights based on the given context.
    # You will be replying to users who may be confused if you don't respond appropriately.
    # You are provided with a PDF document for context.

    Answer the question based only on the following context:
    GentAI, an AI PDF Expert, provides assistance based on the given context.
    To get started, please follow these steps:
    1. Briefly introduce yourself as the AI PDF Expert GentAI.
    2. Describe the content.
    3. Provide 3 example questions using bullet points.
    ---
    {context}

    ---

    Answer the question based on the above context: {question}
    """

#    data = request.get_json()
 #   print(data)
  #  query_text = data['question']

   # print(query_text)

    # Get data and chroma paths dynamically
    data_path, chroma_path = get_data_and_chroma_path_web()
    print(chroma_path)
    data = request.json
    query_text = data['question']
    print(query_text)
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)

    print(db)
    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    print(results)
    if len(results) == 0 or results[0][1] < 0.7: #or #results[0][1] < 0.5:
        # If no matching results found, answer the question directly.
        model = ChatOpenAI()
        response_text = model.predict(query_text)
        response_text = "\n\nI am GentAI, the AI PDF Expert." + response_text

        sources = []  # No sources available since it's not based on context

        return jsonify({"response_text": response_text, "sources": sources})


    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    print(context_text)
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt_kwargs = {
        "context": context_text,
        "question": query_text,
        "answer": ""  # Placeholder for the answer
    }
    prompt = prompt_template.format(**prompt_kwargs)

    model = ChatOpenAI()
    response_text = model.predict(prompt)
    print(response_text)
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = {"response_text": response_text, "sources": sources}
    print(formatted_response)
    #return jsonify(formatted_response)
   # print("data"+user_id)
    # Insert question and answer into ChatLogs table based on user_id
    connection = get_database_connection()  # Assuming you have a function to establish a database connection
    cursor = connection.cursor()
    try:

       token = request.headers.get('Authorization')
       if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
            print("Inside_web"+actual_token)
       else:
            actual_token = None
            response_data = {"message": "Invalid token format"}
            return None, None
       get_user_id_query = 'SELECT user_id FROM Customers WHERE api_token = %s'
       cursor.execute(get_user_id_query, (actual_token,))
       user_id_result = cursor.fetchone()
       user_id=user_id_result[0]
       print(user_id)
       response_string = json.dumps(formatted_response)

       insert_chat_log_query = "INSERT INTO ChatLogs (user_id, question, answer,DateTimeColumn) VALUES (%s, %s, %s,CURRENT_TIMESTAMP)"
       chat_log_data = (user_id, query_text,response_string )
       print("inside+web",chat_log_data)
       cursor.execute(insert_chat_log_query, chat_log_data)
       connection.commit()

    except Exception as e:
        send_error_email('query_web', str(e))
        print("An error occurred:", e)
    finally:
        cursor.close()
        connection.close()
    return jsonify(formatted_response)






from flask import Flask, request, jsonify
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import openai
from multiprocessing import Pool




def pdf_question_answering(user_question, pdf_dir):
    pdf_results = []
    print("From Document:")
    if os.path.exists(pdf_dir) and os.path.isdir(pdf_dir):
        pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
        if not pdf_files:
            print("No PDF files found in the specified directory.")
        else:
            for pdf_file in pdf_files:
                pdf_path = os.path.join(pdf_dir, pdf_file)
                if os.path.exists(pdf_path):
                    pdf_reader = PdfReader(pdf_path)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                    text_splitter = CharacterTextSplitter(
                        separator="\n",
                        chunk_size=1000,
                        chunk_overlap=200,
                        length_function=len
                    )
                    chunks = text_splitter.split_text(text)
                    embeddings = OpenAIEmbeddings()
                    knowledge_base = FAISS.from_texts(chunks, embeddings)
                    docs = knowledge_base.similarity_search(user_question)
                    llm = OpenAI(model_name="gpt-3.5-turbo")
                    chain = load_qa_chain(llm, chain_type="stuff")
                    with get_openai_callback() as cb:
                        response = chain.run(input_documents=docs, question=user_question)
                    if response and not "I don't know" in response:
                        pdf_results.append(f"Response from {pdf_file}:\n{response}")
    return pdf_results

@app.route('/qa', methods=['POST'])
def query():
    connection = get_database_connection()
    cursor = connection.cursor()
    token = request.headers.get('Authorization')
    print("token="+token)
    try:
        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
        else:
            # Handle the case when the token is missing or in the wrong format
            return jsonify({'error': 'Invalid token'}), 401

        # Retrieve user information using the token
        get_user_id_query = 'SELECT user_id FROM Customers WHERE api_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()

        if user_id_result:
            user_id = user_id_result[0]
            print(user_id)
            get_directory_query = 'SELECT directory_name FROM Directory WHERE user_id = %s'
            cursor.execute(get_directory_query, (user_id,))
            directory_info = cursor.fetchone()

            if directory_info:
                directory_name = directory_info[0]
                user_path=os.path.join(BASE_UPLOAD_FOLDER, directory_name)
                user_directory_path = os.path.join(BASE_UPLOAD_FOLDER, directory_name, UPLOAD_FOLDER)
             
            data = request.get_json()
            user_question = data['question']
            print(user_question)
            pdf_dir = user_directory_path
            print(pdf_dir)
            results = pdf_question_answering(user_question, pdf_dir)
            print(results)

            # Convert results from JSON format to string
            results_string = '\n'.join(json.dumps(result) for result in results)

            # Insert question and answer into ChatLogs table based on user_id
            insert_chat_log_query = "INSERT INTO ChatLogs (user_id, question, answer) VALUES (%s, %s, %s)"
            chat_log_data = (user_id, user_question, results_string)
            cursor.execute(insert_chat_log_query, chat_log_data)
            connection.commit()

            cursor.close()
            connection.close()
            return jsonify(results)
        else:
            return jsonify({'error': 'User not found'}), 404
    except Exception as e:
        send_error_email('qa', str(e))
        # Handle any exceptions that occur during the process
        print(f"Error: {e}")
        return jsonify({'error': 'An error occurred while processing the request'}), 500

    



def pdf_question_answering_plugin(user_question, pdf_dir):
    pdf_results = []
    print("From Document:")
    if os.path.exists(pdf_dir) and os.path.isdir(pdf_dir):
        pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
        if not pdf_files:
            print("No PDF files found in the specified directory.")
        else:
            for pdf_file in pdf_files:
                pdf_path = os.path.join(pdf_dir, pdf_file)
                if os.path.exists(pdf_path):
                    pdf_reader = PdfReader(pdf_path)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                    text_splitter = CharacterTextSplitter(
                        separator="\n",
                        chunk_size=1000,
                        chunk_overlap=200,
                        length_function=len
                    )
                    chunks = text_splitter.split_text(text)
                    embeddings = OpenAIEmbeddings()
                    knowledge_base = FAISS.from_texts(chunks, embeddings)
                    docs = knowledge_base.similarity_search(user_question)
                    llm = OpenAI(model_name="gpt-3.5-turbo")
                    chain = load_qa_chain(llm, chain_type="stuff")
                    with get_openai_callback() as cb:
                        response = chain.run(input_documents=docs, question=user_question)
                    if response and not "I don't know" in response:
                        pdf_results.append(f"Response from {pdf_file}:\n{response}")
    return pdf_results

@app.route('/qa_plgin', methods=['POST'])
def query_plugin():
    connection = get_database_connection()
    cursor = connection.cursor()
    user_id = request.headers.get('Authorization')
    print("user_id="+user_id)
    try:
        if user_id and user_id.startswith('Bearer '):
           user_id = user_id.split('Bearer ')[1]
        else:
            # Handle the case when the token is missing or in the wrong format
            return jsonify({'error': 'Invalid token'}), 401

        # Retrieve user information using the token
        
        get_directory_query = 'SELECT directory_name FROM Directory WHERE user_id = %s'
        cursor.execute(get_directory_query, (user_id,))
        directory_info = cursor.fetchone()

        if directory_info:
            directory_name = directory_info[0]
            user_path=os.path.join(BASE_UPLOAD_FOLDER, directory_name)
            user_directory_path = os.path.join(BASE_UPLOAD_FOLDER, directory_name, UPLOAD_FOLDER)

        data = request.get_json()
        user_question = data['question']
        print(user_question)
        pdf_dir = user_directory_path
        print(pdf_dir)
        results = pdf_question_answering_plugin(user_question, pdf_dir)
        print(results)

        # Convert results from JSON format to string
        results_string = '\n'.join(json.dumps(result) for result in results)

        # Insert question and answer into ChatLogs table based on user_id
        insert_chat_log_query = "INSERT INTO ChatLogs (user_id, question, answer) VALUES (%s, %s, %s)"
        chat_log_data = (user_id, user_question, results_string)
        cursor.execute(insert_chat_log_query, chat_log_data)
        connection.commit()

        cursor.close()
        connection.close()
        return jsonify(results)
     
    except Exception as e:
        send_error_email('qa_plugin', str(e))
        # Handle any exceptions that occur during the process
        print(f"Error: {e}")
        return jsonify({'error': 'An error occurred while processing the request'}), 500

    



@app.route('/login', methods=['POST'])
def login():
    data = request.json

    email = data.get('email')
    password = data.get('password')

    # Check if required fields are present
    if not email or not password:
        return jsonify({'error': 'Missing required fields'}), 400

    try:
        # Initialize cursor after establishing connection
        connection = get_database_connection()
        cursor = connection.cursor()

        # Check if the user with the given email exists
        check_email_query = 'SELECT user_id, password, first_name FROM Customers WHERE email = %s'
        cursor.execute(check_email_query, (email,))
        user = cursor.fetchone()

        if not user:
            cursor.close()
            return jsonify({'error': 'Invalid email or password'}), 401

        user_id, hashed_password, user_first_name = user

        # Check if the password is correct
        if not check_password_hash(hashed_password, password):
            cursor.close()
            return jsonify({'error': 'Invalid email or password'}), 401

        # Generate a unique api_token for the user
        api_token = secrets.token_hex(16)

        # Update the user with the generated api_token
        update_token_query = 'UPDATE Customers SET api_token = %s WHERE email = %s'
        cursor.execute(update_token_query, (api_token, email))
        connection.commit()

        # Retrieve the user's directory name from the Directory table
        get_directory_query = 'SELECT directory_name FROM Directory WHERE user_id = %s'
        cursor.execute(get_directory_query, (user_id,))
        directory_info = cursor.fetchone()
        user_token_query = 'SELECT user_token FROM Customers WHERE user_id = %s'
        cursor.execute(user_token_query, (user_id,))
        user_token = cursor.fetchone()

        if directory_info:
            directory_name = directory_info[0]
            user_directory_path = os.path.join(BASE_UPLOAD_FOLDER, directory_name)

            context_file_path = os.path.join(user_directory_path, CONTEXT_FILE)
            if os.path.exists(context_file_path):
                with open(context_file_path) as file:
                    context = json.load(file)

            # Load user-specific sources from the sources file
            sources_file_path = os.path.join(user_directory_path, SOURCES_FILE)
            if os.path.exists(sources_file_path):
                with open(sources_file_path) as sources_file:
                    context["sources"] = list(map(lambda e: e.strip(), sources_file.readlines()))
                    context["collection_exists"] = True


            cursor.close()
            connection.close()

            return jsonify({'api_token': api_token,'user_token':user_token, 'message': 'Login successful', 'redirect': False}), 200
        else:
            cursor.close()
            connection.close()
            return jsonify({'error': 'User directory not found'}), 404

    except Exception as e:
        send_error_email('login', str(e))
        print(f"Error: {e}")
        return jsonify({'error': 'An error occurred while processing the request'}), 500


@app.route('/login_expired', methods=['POST'])
def login_expired():
    try:
        connection = get_database_connection()
        cursor = connection.cursor()
        token = request.headers.get('Authorization')
        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
            print("token"+actual_token)
            token_user = request.json.get('token')
            print("token="+token_user)
            check_token_query = 'SELECT user_token FROM Customers WHERE user_token = %s'
            cursor.execute(check_token_query, (token_user,))
        
            token_exists = cursor.fetchone()
            print(token_exists)
            token_exists=token_exists[0]

            if token_exists:
                # Token exists, return success message along with the token
                return jsonify({'message': 'Token exists', 'api_token': token_exists}), 200
            else:
                # Token does not exist
                return jsonify({'error': 'Invalid token'}), 401
        else:
            return jsonify({'error': 'Token missing or invalid'}), 401
    except Exception as e:
        send_error_email('login_expired', str(e))
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        connection.close()

def generate_random_password(length=8):
    # Define the characters to choose from
    characters = string.ascii_letters + string.digits
    
    # Generate a random password by choosing characters randomly
    password = ''.join(random.choice(characters) for _ in range(length))
    print(password)
    
    return password



@app.route('/forgot_password', methods=['POST'])
def forgot_password():
    try:
        data = request.json
        email = data.get('email')
        print(email)

        # Get database connection
        connection = get_database_connection()
        cursor = connection.cursor()

        # Check if the email exists in the database
        check_email_query = 'SELECT user_id FROM Customers WHERE email = %s'
        cursor.execute(check_email_query, (email,))
        user_data = cursor.fetchone()
        if user_data is None:
            return jsonify({'error': 'Email not found'}), 404

        user_id = user_data[0]

        # Generate a random password
        new_password = generate_random_password()
        print(new_password)

        # Check if the password timestamp is within the last 4 hours
        # (You need to implement this check)

        # Hash the new password
        hashed_password = generate_password_hash(new_password, method='pbkdf2:sha256')
        print(hashed_password)

        # Update the user's password and timestamp in the database
        update_password_query = 'UPDATE Customers SET password = %s WHERE user_id = %s'
        cursor.execute(update_password_query, (hashed_password, user_id))
        connection.commit()  # Commit changes to the database

        send_email_password(email, new_password)

        return jsonify({'message': 'Password reset email sent successfully'}), 200
    except Exception as e:
#        send_error_email('forgot_password', str(e))
        # Handle exceptions
        return jsonify({'error': str(e)}), 500
    finally:
        # Close cursor and connection in finally block
        cursor.close()
        connection.close()

def send_email_password(recipient, password):
    subject = "Temporary Password"
    msg = Message(subject, recipients=[recipient])
    msg.body = f"Here is your email and password:\n\nUser Email: {recipient}\nPassword: {password}"
    mail.send(msg)

from flask import request, jsonify
from werkzeug.security import generate_password_hash
from datetime import datetime, timedelta

@app.route('/get_password', methods=['POST'])
def get_password():
    try:
        data = request.json
        email = data.get('email')
        print(email)

        # Get database connection
        connection = get_database_connection()
        cursor = connection.cursor()

        # Check if the email exists in the database
        check_email_query = 'SELECT user_id, password FROM Customers WHERE email = %s'
        cursor.execute(check_email_query, (email,))
        user_data = cursor.fetchone()
        if user_data is None:
            return jsonify({'error': 'Email not found'}), 404

        user_id, hashed_password = user_data

        # Generate a random password
        new_password = generate_random_password()
        print(new_password)

        # Hash the new password
        hashed_new_password = generate_password_hash(new_password, method='pbkdf2:sha256')

        # Update the user's password in the database (optional)
        update_password_query = 'UPDATE Customers SET password = %s WHERE user_id = %s'
        cursor.execute(update_password_query, (hashed_new_password, user_id))
        connection.commit()  # Commit changes to the database

        # Send the password to the user's email
        send_email_password(email, new_password)

        return jsonify({'message': 'Password sent successfully to your email'}), 200
    except Exception as e:
        send_error_email('get_password', str(e))
        # Handle exceptions
        return jsonify({'error': str(e)}), 500
    finally:
        # Close cursor and connection in finally block
        if cursor:
            cursor.close()
        if connection:
            connection.close()

def validate_password(password):
    # Implement your password validation logic
    # For example, check if the password meets certain criteria
    return len(password) >= 8  # Example: Password should be at least 8 characters long

@app.route('/reset-password', methods=['POST'])
def reset_password():
    try:
        connection = get_database_connection()
        cursor = connection.cursor()

        data = request.json
        email = data.get('email')
        old_password = data.get('old_password')
        new_password = data.get('new_password')

        # Check if required fields are present
        if not email or not old_password or not new_password:
            return jsonify({'error': 'Missing required fields'}), 400

        # Check if the email exists in the database
        check_email_query = 'SELECT user_id, password FROM Customers WHERE email = %s'
        cursor.execute(check_email_query, (email,))
        user_data = cursor.fetchone()

        if user_data:
            user_id, hashed_current_password = user_data
            
            # Check if the old password matches the current password
            if not check_password_hash(hashed_current_password, old_password):
                return jsonify({'error': 'Old password is incorrect'}), 400

            # Ensure the new password is not the same as the old password
            if old_password == new_password:
                return jsonify({'error': 'New password cannot be the same as the old password'}), 400

            # Validate the new password
            if not validate_password(new_password):
                return jsonify({'error': 'Invalid password format. Password must be at least 8 characters long'}), 400

           
            hashed_password = generate_password_hash(new_password, method='pbkdf2:sha256')

            # Update the user's password in the database
            update_password_query = 'UPDATE Customers SET password = %s WHERE user_id = %s'
            cursor.execute(update_password_query, (hashed_password, user_id))
            connection.commit()

            # Send a password reset email
            msg = Message('Password Reset', recipients=[email])
            msg.body = f'Your password has been reset successfully. Password:\n\n{new_password}'
            mail.send(msg)

            return jsonify({'message': 'Your password has been reset successfully and an email has been sent with the new password'}), 200
        else:
            return jsonify({'error': 'Email not found'}), 404
    except Exception as e:
        send_error_email('reset-password', str(e))
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        connection.close()


from functools import wraps
import jwt  # I


from flask import send_from_directory, make_response

@app.route('/get_images/<image_name>', methods=['GET'])
def get_image(image_name):
    try:
        token = request.headers.get('Authorization')
        print("Token:", token)

        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
            print(actual_token)

            # Retrieve user_id and directory name from the database using token
            connection = get_database_connection()  # Assuming you have a function to get the database connection
            cursor = connection.cursor()
            cursor.execute('SELECT user_id FROM Customers WHERE api_token = %s', (actual_token,))
            user_id_result = cursor.fetchone()

            if user_id_result:
                user_id = user_id_result[0]
                print("User ID:", user_id)

                cursor.execute('SELECT directory_name FROM Directory WHERE user_id = %s', (user_id,))
                directory_info = cursor.fetchone()
                print(directory_info)

                if directory_info:
                    directory_name = directory_info[0]
                    print("Directory:", directory_name)

                    image_directory = os.path.join(BASE_UPLOAD_FOLDER, directory_name, "image")

                    if os.path.exists(image_directory):
                        response = make_response(send_from_directory(image_directory, image_name))
                        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
                        response.headers['Pragma'] = 'no-cache'
                        response.headers['Expires'] = '0'
                        return response
                    else:
                        return jsonify({'error': 'Image directory not found'}), 404
                else:
                    return jsonify({'error': 'Directory not found for the user'}), 404
            else:
                return jsonify({'error': 'User not found'}), 404
        else:
            return jsonify({'error': 'Invalid token'}), 401
    except Exception as e:
        send_error_email('get_images/<image_name>', str(e))
        return jsonify({'error': str(e)}), 500
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


ALLOWED_EXTEN = {'png', 'jpg', 'jpeg', 'gif'}


@app.route('/upload_image', methods=['POST'])
def upload_image():
    try:
        token = request.headers.get('Authorization')
        print("Token:", token)

        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
            print(actual_token)

            # Retrieve user_id and directory name from the database using token
            connection = get_database_connection()
            cursor = connection.cursor()
            cursor.execute('SELECT user_id FROM Customers WHERE api_token = %s', (actual_token,))
            user_id_result = cursor.fetchone()

            if user_id_result:
                user_id = user_id_result[0]
                print("User ID:", user_id)

                cursor.execute('SELECT directory_name FROM Directory WHERE user_id = %s', (user_id,))
                directory_info = cursor.fetchone()
                print(directory_info)

                if directory_info:
                    directory_name = directory_info[0]
                    print("Directory:", directory_name)

                    # Check if the post request has the file part
                    if 'file' not in request.files:
                        return jsonify({'error': 'No file part in the request'}), 400

                    file = request.files['file']

                    # Check if the file is one of the allowed types/extensions
                    if file and allowed_filename(file.filename):

                        save_folder = os.path.join(BASE_UPLOAD_FOLDER, directory_name, "image")
                        if not os.path.exists(save_folder):
                            os.makedirs(save_folder)

                        # Remove old images if they exist (removed as per your request)

                        # Save the received image file to the specified directory
                        # Use a single name for the image file
                        filename = "uploaded_image.jpg"  # You can choose any desired name
                        file.save(os.path.join(save_folder, filename))
                        return jsonify({'message': 'Image uploaded successfully'})
                    else:
                        return jsonify({'error': 'Invalid file type. Allowed file types are: png, jpg, jpeg, gif'}), 400
                else:
                    return jsonify({'error': 'Directory not found for the user'}), 404
            else:
                return jsonify({'error': 'User not found'}), 404
        else:
            return jsonify({'error': 'Invalid token'}), 401
    except Exception as e:
        send_error_email('upload_image', str(e))
        return jsonify({'error': str(e)}), 500
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


def allowed_filename(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTEN














ALLOWED_EXTENSIONS = {'pdf'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

# Function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to get file size
def get_file_size(file):
    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)
    return size

@app.route("/include_source", methods=['POST'])
def include_source():
    response_data = {"success": False, "message": ""}
    try:
        file = request.files.get('file')
        include_url = request.form.get('include-url')

        if file:
            # Check if file is allowed
            if not allowed_file(file.filename):
                response_data["message"] = "Only PDF files are allowed"
                print(response_data)
                return jsonify(response_data)

            # Check file size
            if get_file_size(file) > MAX_FILE_SIZE:
                response_data["message"] = "File size exceeds maximum limit (10 MB)"
                return jsonify(response_data)

        # Initialize cursor after establishing connection
        connection = get_database_connection()
        cursor = connection.cursor()

        token = request.headers.get('Authorization')

        # Check if the token starts with 'Bearer ' and extract the actual token
        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
        else:
            # Handle the case when the token is not in the expected format
            actual_token = None
            response_data["message"] = "Invalid token format"
            cursor.close()
            connection.close()
            return jsonify(response_data)

        # Retrieve the user's ID from the Customers table based on the token
        get_user_id_query = 'SELECT user_id, first_name FROM Customers WHERE api_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()

        if user_id_result:
            user_id, first_name = user_id_result
            user_folder_name = f"{first_name}_{user_id}"

            # Retrieve the user's directory name from the Directory table
            get_directory_query = 'SELECT directory_name FROM Directory WHERE user_id = %s'
            cursor.execute(get_directory_query, (user_id,))
            directory_info = cursor.fetchone()

            if directory_info:
                directory_name = directory_info[0]
                user_directory_path = os.path.join(BASE_UPLOAD_FOLDER, directory_name, UPLOAD_FOLDER)

                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    path = os.path.join(user_directory_path, filename)
                    file.save(path)
                    insert_filename_query = 'INSERT INTO FileLogs (user_id, file_name) VALUES (%s, %s) ON DUPLICATE KEY UPDATE user_id = VALUES(user_id), file_name = VALUES(file_name)'

                    cursor.execute(insert_filename_query, (user_id, filename))
                    connection.commit()  # Commit the transaction

                    context["sources_to_add"].append(filename)

                    response_data["success"] = True
                    response_data["message"] = "File uploaded successfully"

                elif include_url:
                    save_directory = user_directory_path
                    print(save_directory)

                    crawl_website(include_url, save_directory)

                    url = include_url
                    insert_url_query = 'INSERT INTO WebURLlog (user_id,url) VALUES (%s, %s) ON DUPLICATE KEY UPDATE user_id = VALUES(user_id), url = VALUES(url)'

                    cursor.execute(insert_url_query, (user_id, url))
                    connection.commit()

                    context["sources_to_add"].append(include_url)
                    response_data["success"] = True
                    response_data["message"] = "URL added successfully"
                else:
                    response_data["message"] = "Invalid file or URL"
            else:
                response_data["message"] = "User directory not found"
        else:
            response_data["message"] = "Invalid token"

        cursor.close()
        connection.close()

    except Exception as e:
        send_error_email('include_source', str(e))
        print(f"Error: {e}")
        response_data["message"] = "An error occurred while processing the request"

    return jsonify(response_data)



@app.route('/delete_urls', methods=['POST'])
def delete_urls():
    try:
        token = request.headers.get('Authorization')
        print("Token:", token)

        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
            print(actual_token)

            # Retrieve user_id and directory name from the database using token
            connection = get_database_connection()
            cursor = connection.cursor()
            cursor.execute('SELECT user_id FROM Customers WHERE api_token = %s', (actual_token,))
            user_id_result = cursor.fetchone()
            if user_id_result:
                user_id = user_id_result[0]
                print("User ID:", user_id)

                cursor.execute('SELECT directory_name FROM Directory WHERE user_id = %s', (user_id,))
                directory_info = cursor.fetchone()
                if directory_info:
                    directory_name = directory_info[0]
                    print("Directory:", directory_name)

                    # Get the filenames to delete from the request JSON
                    data = request.json
                    print(data)
                    print("Request Data:", data)
                    if 'urls' in data:
                        filenames = data['urls']
                        submenu_id = data.get('submenu_id')
                        print("Filenames:", filenames)
                        for filename in filenames: 
                            url=filename# Corrected variable name
                            print(filename)
                            filename = filename.replace('://', '___').replace('/', '-').replace('.', '_') + ".pdf"
                            print(filename)
                            file_path = os.path.join(BASE_UPLOAD_FOLDER, directory_name)

                            # Check if file exists and delete it
                            if os.path.exists(file_path) and os.path.isdir(file_path):
                               for root, dirs, files in os.walk(file_path):
                                   for file in files:
                                       if file == filename:
                                           os.remove(os.path.join(root, file))
                                # Delete URLs from WebURLlog table for the given user_id and filenames
                        chroma_path = os.path.join(BASE_UPLOAD_FOLDER, directory_name, f"service{submenu_id}_chromadb")
                        if os.path.exists(chroma_path):
                            shutil.rmtree(chroma_path)
                            print("Deleted ChromaDB folder:", chroma_path)

                        upsert_file_query = '''
                                INSERT INTO url_reports (url, deleted, deleted_date, status, user_id)
                                VALUES (%s, 'No', NULL, 'Enabled', %s)
                                ON DUPLICATE KEY UPDATE deleted = 'Yes', deleted_date = CURRENT_TIMESTAMP, status = 'Disabled'
                            '''
                        cursor.execute(upsert_file_query, (url, user_id))


                        # Commit the database changes
                        connection.commit()

                        # Close cursor and connection
                        cursor.close()
                        connection.close()

                        return jsonify({'message': 'Urls deleted successfully'}), 200
                    else:
                        return jsonify({'error': 'Urls not provided in request'}), 400
                else:
                    return jsonify({'error': 'Directory not found for the user'}), 404
            else:
                return jsonify({'error': 'User not found'}), 404
        else:
            return jsonify({'error': 'Invalid token'}), 401
    except Exception as e:
        send_error_email('delete_urls', str(e))
        return jsonify({'error': str(e)}), 500








@app.route('/save_settingurls', methods=['POST'])
def save_settingurls():
    connection = None
    cursor = None
    try:
        connection = get_database_connection()
        cursor = connection.cursor()
        token = request.headers.get('Authorization')
        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
        else:
            return jsonify({'error': 'Invalid token'}), 401

        get_user_id_query = 'SELECT user_id FROM Customers WHERE api_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()

        if user_id_result:
            user_id = user_id_result[0]
            data = request.json
            urls = data.get('urls', [])

            # Iterate over the urls and perform an update query for each URL
            for url_data in urls:
                update_query = """
                INSERT INTO setting_urls (user_id, url1, url2, url3, description1, description2, description3,Title1,Title2,Title3)
                VALUES (%s, %s, %s, %s, %s, %s, %s,%s,%s,%s)
                ON DUPLICATE KEY UPDATE url1 = VALUES(url1), url2 = VALUES(url2), url3 = VALUES(url3),
                description1 = VALUES(description1), description2 = VALUES(description2), description3 = VALUES(description3),Title1=VALUES(Title1),Title2=VALUES(Title2),Title3=VALUES(Title3);
                """
                # Extracting data for each URL from url_data dictionary
                url1 = url_data.get('url1')
                url2 = url_data.get('url2')
                url3 = url_data.get('url3')
                description1 = url_data.get('description1')
                description2 = url_data.get('description2')
                description3 = url_data.get('description3')
                Title1=url_data.get('title1')
                Title2=url_data.get('title2')
                Title3=url_data.get('title3')

                print(url1)
                print(url2)
                print(url3)
                print(description1)
                print(description2)
                print(description3)
                # Executing the query with the extracted data
                cursor.execute(update_query, (user_id, url1, url2, url3, description1, description2, description3,Title1,Title2,Title3))

            connection.commit()  # Commit the transaction after all URLs are processed
            return jsonify({'message': 'URLs saved successfully'}), 200
        else:
            return jsonify({'error': 'User not found'}), 404

    except Exception as e:
        send_error_email('save_settingurls', str(e))
        connection.rollback()  # Rollback the transaction in case of an error
        return jsonify({'error': str(e)}), 500
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


@app.route('/get_settingurls', methods=['GET'])
def get_settingurls():
    connection = None
    cursor = None
    try:
        connection = get_database_connection()
        cursor = connection.cursor()
        token = request.headers.get('Authorization')
        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
        else:
            return jsonify({'error': 'Invalid token'}), 401

        get_user_id_query = 'SELECT user_id FROM Customers WHERE user_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()

        if user_id_result:
            user_id = user_id_result[0]

            # Retrieve all URLs for the user
            get_urls_query = """
            SELECT url1, url2, url3, description1, description2, description3, Title1, Title2, Title3
            FROM setting_urls
            WHERE user_id = %s;
            """
            cursor.execute(get_urls_query, (user_id,))
            urls_result = cursor.fetchone()

            if urls_result:
                urls = {
                    'url1': urls_result[0],
                    'url2': urls_result[1],
                    'url3': urls_result[2],
                    'description1': urls_result[3],
                    'description2': urls_result[4],
                    'description3': urls_result[5],
                    'Title1': urls_result[6],
                    'Title2': urls_result[7],
                    'Title3': urls_result[8]
                }
                print(urls)
                return jsonify(urls), 200
            else:
                return jsonify({'message': 'No URLs found for the user'}), 404
        else:
            return jsonify({'error': 'User not found'}), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()





@app.route('/save_eventurls', methods=['POST'])
def save_eventurls():
    connection = None
    cursor = None
    try:
        connection = get_database_connection()
        cursor = connection.cursor()
        token = request.headers.get('Authorization')
        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
        else:
            return jsonify({'error': 'Invalid token'}), 401

        get_user_id_query = 'SELECT user_id FROM Customers WHERE api_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()

        if user_id_result:
            user_id = user_id_result[0]
            data = request.json
            urls = data.get('urls', [])

            # Iterate over the urls and perform an insert or update query for each URL
            for url_data in urls:
                insert_query = """
                INSERT INTO setting_event (user_id, url1, url2, url3, description1, description2, description3, title1, title2, title3)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE url1 = VALUES(url1), url2 = VALUES(url2), url3 = VALUES(url3),
                description1 = VALUES(description1), description2 = VALUES(description2), description3 = VALUES(description3),
                title1 = VALUES(title1), title2 = VALUES(title2), title3 = VALUES(title3);
                """
                # Extracting data for each URL from url_data dictionary
                url1 = url_data.get('url1')
                url2 = url_data.get('url2')
                url3 = url_data.get('url3')
                description1 = url_data.get('description1')
                description2 = url_data.get('description2')
                description3 = url_data.get('description3')
                title1 = url_data.get('title1')
                title2 = url_data.get('title2')
                title3 = url_data.get('title3')

                # Executing the query with the extracted data
                cursor.execute(insert_query, (user_id, url1, url2, url3, description1, description2, description3, title1, title2, title3))

            connection.commit()  # Commit the transaction after all URLs are processed
            return jsonify({'message': 'URLs saved successfully'}), 200
        else:
            return jsonify({'error': 'User not found'}), 404

    except Exception as e:
        send_error_email('save_eventurls', str(e))
        connection.rollback()  # Rollback the transaction in case of an error
        return jsonify({'error': str(e)}), 500
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

@app.route('/get_eventurls', methods=['GET'])
def get_eventurls():
    connection = None
    cursor = None
    try:
        connection = get_database_connection()
        cursor = connection.cursor()
        token = request.headers.get('Authorization')
        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
        else:
            return jsonify({'error': 'Invalid token'}), 401

        get_user_id_query = 'SELECT user_id FROM Customers WHERE api_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()

        if user_id_result:
            user_id = user_id_result[0]

            # Retrieve all URLs for the user
            get_urls_query = """
            SELECT url1, url2, url3, description1, description2, description3, title1, title2, title3
            FROM setting_event
            WHERE user_id = %s;
            """
            cursor.execute(get_urls_query, (user_id,))
            urls_result = cursor.fetchone()

            if urls_result:
                urls = {
                    'url1': urls_result[0],
                    'url2': urls_result[1],
                    'url3': urls_result[2],
                    'description1': urls_result[3],
                    'description2': urls_result[4],
                    'description3': urls_result[5],
                    'title1': urls_result[6],
                    'title2': urls_result[7],
                    'title3': urls_result[8]
                }
                print(urls)
                return jsonify(urls), 200
            else:
                return jsonify({'message': 'No URLs found for the user'}), 404
        else:
            return jsonify({'error': 'User not found'}), 404

    except Exception as e:
        send_error_email('get_eventurls', str(e))
        return jsonify({'error': str(e)}), 500
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

MAX_IMAGES = 3  # Maximum number of images to keep in the directory

@app.route('/upload_setting_image', methods=['POST'])
def upload_setting_image():
    try:
        token = request.headers.get('Authorization')
        print("Token:", token)

        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
            print(actual_token)

            # Retrieve user_id and directory name from the database using token
            connection = get_database_connection()
            cursor = connection.cursor()
            cursor.execute('SELECT user_id FROM Customers WHERE api_token = %s', (actual_token,))
            user_id_result = cursor.fetchone()

            if user_id_result:
                user_id = user_id_result[0]
                print("User ID:", user_id)

                cursor.execute('SELECT directory_name FROM Directory WHERE user_id = %s', (user_id,))
                directory_info = cursor.fetchone()
                print(directory_info)

                if directory_info:
                    directory_name = directory_info[0]
                    print("Directory:", directory_name)

                    # Check if the post request has the file part
                    if 'file' not in request.files:
                        return jsonify({'error': 'No file part in the request'}), 400

                    # Retrieve files from the request
                    files = request.files.getlist('file')

                    # Check if there are any files
                    if not files:
                        return jsonify({'error': 'No file uploaded'}), 400
                    # Check if the file is one of the allowed types/extensions
                    for file in files:
                        filename = secure_filename(file.filename)
                        ext = os.path.splitext(filename)[1].lower()
                        if ext not in ['.png', '.jpg', '.jpeg', '.gif']:
                           return jsonify({'error': 'Invalid file type. Allowed file types are: png, jpg, jpeg, gif'}), 400

                    # Iterate over files

                    save_folder = os.path.join(BASE_UPLOAD_FOLDER, directory_name, "setting_image")
                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)

                    # Remove old images if they exist
                    for i in range(1, MAX_IMAGES + 1):
                        old_image_path = os.path.join(save_folder, f"image{i}{ext}")
                        if os.path.exists(old_image_path):
                            os.remove(old_image_path)

                    # Save the received image files to the specified directory
                    # Use a single name for each image file
                    for index, file in enumerate(files, start=1):
                        filename = f"image{index}{ext}"
                        file.save(os.path.join(save_folder, filename))

                    return jsonify({'message': 'Images uploaded successfully'})
                else:
                    return jsonify({'error': 'Directory not found for the user'}), 404
            else:
                return jsonify({'error': 'User not found'}), 404
        else:
            return jsonify({'error': 'Invalid token'}), 401
    except Exception as e:
        send_error_email('upload_setting_image', str(e))
        return jsonify({'error': str(e)}), 500
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()





def get_image_format(file_name):
    _, extension = os.path.splitext(file_name)
    if extension.lower() == '.jpg' or extension.lower() == '.jpeg':
        return 'jpeg'
    elif extension.lower() == '.png':
        return 'png'
    elif extension.lower() == '.gif':
        return 'gif'
    # Add more image formats as needed
    else:
        return None  # Unknown format
from flask import Flask, request, jsonify, Response  # Add this import statement

@app.route('/get_all_setting_images', methods=['GET'])
def get_all_setting_images():
    cursor = None
    connection = None
    try:
        token = request.headers.get('Authorization')

        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
            connection = get_database_connection()
            cursor = connection.cursor()
            cursor.execute('SELECT user_id FROM Customers WHERE api_token = %s', (actual_token,))
            user_id_result = cursor.fetchone()
            if user_id_result:
                user_id = user_id_result[0]
                cursor.execute('SELECT directory_name FROM Directory WHERE user_id = %s', (user_id,))
                directory_info = cursor.fetchone()
                if directory_info:
                    directory_name = directory_info[0]
                    save_folder = os.path.join(BASE_UPLOAD_FOLDER, directory_name, "setting_image")

                    # Get list of all files in the directory
                    files = os.listdir(save_folder)


            # Retrieve user_id and directory name from the database using token
            # Assuming you have functions to handle database operations
                    # Get list of all files in the directory

                    # Filter out non-image files
                    image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]

                    # Prepare a list to store image data
                    image_data = []

                    # Read each image file and append its binary data to the list
                    for image_file in image_files:
                        with open(os.path.join(save_folder, image_file), 'rb') as file:
                            image_data.append(file.read())

                    # Get the image format for the first image file
                    image_format = get_image_format(image_files[0]) if image_files else None
                    if image_format:
                        # Return the list of image data as response with appropriate mimetype
                        return Response(response=image_data, status=200, mimetype=f'image/{image_format}')
                    else:
                        return jsonify({'error': 'Unknown image format'}), 500
                else:
                    return jsonify({'error': 'Directory not found for the user'}), 404
            else:
                return jsonify({'error': 'User not found'}), 404
        else:
            return jsonify({'error': 'Invalid token'}), 401
    except Exception as e:
        send_error_email('get_all_setting_images', str(e))
        print(f"Error in get_all_setting_images: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

@app.route('/event_settingurls', methods=['POST'])
def event_settingurls():
    cursor = None
    try:
        connection = get_database_connection()
        cursor = connection.cursor()
        token = request.headers.get('Authorization')
        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
        else:
            return jsonify({'error': 'Invalid token'}), 401

        get_user_id_query = 'SELECT user_id FROM Customers WHERE api_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()

        if user_id_result:
            user_id = user_id_result[0]

            # Retrieve user directory information from the database
            get_directory_query = 'SELECT directory_name FROM Directory WHERE user_id = %s'
            cursor.execute(get_directory_query, (user_id,))
            directory_info = cursor.fetchone()

            if not directory_info:
                return jsonify({'error': 'Directory not found for the user'}), 404

            directory_name = directory_info[0]
            user_directory_path = os.path.join(BASE_UPLOAD_FOLDER, directory_name, "setting_image")

            if not os.path.exists(user_directory_path):
                os.makedirs(user_directory_path)

            data = request.json.get('urls')  # Extract data from the JSON request

            update_query = """
                INSERT INTO setting_event (user_id, url1, url2, url3, description1, description2, description3,
                                           title1, title2, title3, image1, image2, image3)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    url1 = VALUES(url1),
                    url2 = VALUES(url2),
                    url3 = VALUES(url3),
                    description1 = VALUES(description1),
                    description2 = VALUES(description2),
                    description3 = VALUES(description3),
                    title1 = VALUES(title1),
                    title2 = VALUES(title2),
                    title3 = VALUES(title3),
                    image1 = VALUES(image1),
                    image2 = VALUES(image2),
                    image3 = VALUES(image3);
            """
            cursor.execute(update_query, (
                user_id,
                data.get('url1'),
                data.get('url2'),
                data.get('url3'),
                data.get('description1'),
                data.get('description2'),
                data.get('description3'),
                data.get('title1'),
                data.get('title2'),
                data.get('title3'),
                data.get('image1'),  # Save image1 filename to database
                data.get('image2'),  # Save image2 filename to database
                data.get('image3')   # Save image3 filename to database
            ))

            # Save images to respective directories
            for i in range(1, 4):
                image_key = f'image{i}'
                if image_key in data:
                    image_data = data[image_key]
                    if image_data["filename"] == f'image{i}' and imghdr.what(None, h=image_data['data']) in ['jpeg', 'png', 'gif']:  # Ensure the file name and extension are correct
                        filename = os.path.join(user_directory_path, f'{image_data["filename"]}.{imghdr.what(None, h=image_data["data"])}')
                        with open(filename,                        'wb') as f:
                            f.write(image_data['data'])
                    else:
                        return jsonify({'error': f'Invalid image data for {image_key}'}), 400

            connection.commit()  # Commit the transaction after all URLs are processed
            return jsonify({'message': 'URLs and images saved successfully'}), 200
        else:
            return jsonify({'error': 'User not found'}), 404

    except Exception as e:
        send_error_email('get_all_setting_images', str(e))
        return jsonify({'error': str(e)}), 500
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()





import base64

def download_image(url, directory, filename):
    filepath = os.path.join(directory, filename)
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                f.write(response.content)
            return filepath
        else:
            print(f"Failed to download image from {url}. Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error downloading image from {url}: {str(e)}")
        return None
@app.route('/get_event', methods=['GET'])
def get_event():
    try:
        connection = get_database_connection()
        cursor = connection.cursor()
        
        token = request.headers.get('Authorization')
        if not token or not token.startswith('Bearer '):
            return jsonify({'error': 'Invalid token'}), 401

        actual_token = token.split('Bearer ')[1]

        get_user_id_query = 'SELECT user_id FROM Customers WHERE api_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()

        if not user_id_result:
            return jsonify({'error': 'User not found'}), 404

        user_id = user_id_result[0]

        get_directory_query = 'SELECT directory_name FROM Directory WHERE user_id = %s'
        cursor.execute(get_directory_query, (user_id,))
        directory_info = cursor.fetchone()

        if not directory_info:
            return jsonify({'message': 'Directory not found for the user'}), 404

        directory_name = directory_info[0]
        user_directory_path = os.path.join(BASE_UPLOAD_FOLDER, directory_name, "setting_image")

        if not os.path.exists(user_directory_path):
            os.makedirs(user_directory_path)

        get_data_query = """
            SELECT url1, url2, url3, description1, description2, description3,
                   title1, title2, title3, image1, image2, image3
            FROM setting_event
            WHERE user_id = %s
        """
        cursor.execute(get_data_query, (user_id,))
        data_result = cursor.fetchone()

        if not data_result:
            return jsonify({'message': 'No data found for the user'}), 404

        data = {
            'url1': data_result[0],
            'url2': data_result[1],
            'url3': data_result[2],
            'description1': data_result[3],
            'description2': data_result[4],
            'description3': data_result[5],
            'title1': data_result[6],
            'title2': data_result[7],
            'title3': data_result[8],
        }

        # Download and save images
        image_urls = [data_result[i] for i in range(9, 12)]
        image_paths = []
        for i, url in enumerate(image_urls, start=1):
            if url:
                filename = f'image{i}'
                image_path = download_image(url, user_directory_path, filename)
                if image_path:
                    image_paths.append(image_path)
                else:
                    image_paths.append(None)
            else:
                image_paths.append(None)
        data['image_paths'] = image_paths  # Include image paths in the response

        return jsonify(data), 200

    except Exception as e:
        send_error_email('get_event', str(e))
        return jsonify({'error': str(e)}), 500

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()
 




import requests
from bs4 import BeautifulSoup
import pdfkit
import time
from urllib.parse import urljoin, urlparse
import os
import re


def crawl_website(url, save_directory):
    try:
        # Set a custom user agent
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

        # Send a GET request to the URL with custom headers
        response = requests.get(url, headers=headers)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the HTML content of the page
            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract the text content from the webpage
            text_content = soup.get_text()

            # Process the text content
            processed_text_content = process_text_content(text_content)

            # Save the PDF for the current page
            save_to_pdf(processed_text_content, url, save_directory)
        else:
            print("Failed to retrieve the webpage. Status code:", response.status_code)
    except Exception as e:
        send_error_email('crawl_website', str(e))

        print("An error occurred:", str(e))

def process_text_content(text_content):
    # Split the text into paragraphs
    paragraphs = text_content.split('\n\n')
    
    # Process each paragraph
    processed_paragraphs = []
    for paragraph in paragraphs:
        # Split the paragraph into sentences
        sentences = re.split(r'(?<=[.!?]) +', paragraph)
        
        # Join the sentences with proper formatting
        processed_paragraph = ' '.join(sentences)
        
        # Append the processed paragraph to the list
        processed_paragraphs.append(processed_paragraph)
    
    # Join the processed paragraphs with proper formatting
    processed_text = '\n\n'.join(processed_paragraphs)

    return processed_text

def save_to_pdf(text_content, url, save_directory):
    try:
        # Specify the path to wkhtmltopdf
        path_to_wkhtmltopdf = '/usr/bin/wkhtmltopdf'  # Specify the path to wkhtmltopdf

        # Generate filename from URL
        filename = url.replace('/', '_').replace(':', '_').replace('.', '_') + ".pdf"

        # Construct the full path to save the PDF
        filepath = os.path.join(save_directory, filename)

        # Convert the text content to PDF using pdfkit
        pdfkit.from_string(text_content, filepath, configuration=pdfkit.configuration(wkhtmltopdf=path_to_wkhtmltopdf))
        print("PDF saved successfully as", filepath)
    except Exception as e:
        print("An error occurred while saving the PDF:", str(e))




@app.route("/add_sources", methods=['GET', 'POST'])
def add_sources():

    token = request.headers.get('Authorization')
    print(token)

    # Check if the token starts with 'Bearer ' and extract the actual token
    if token and token.startswith('Bearer '):
        actual_token = token.split('Bearer ')[1]
        g.api_token = actual_token
        print(g.api_token)
        get_user_id_query = 'SELECT user_id,first_name FROM Customers WHERE api_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()

        if user_id_result:
            user_id = user_id_result[0]
            first_name=user_id_result[1]
            user_folder_name = f"{first_name}_{user_id}"

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
                user_directory = os.path.join(BASE_UPLOAD_FOLDER, directory_name,UPLOAD_FOLDER)
                if context["sources_to_add"]:
                        valid_sources = []

                        for source in context["sources_to_add"]:
                            if validators.url(source) or os.path.exists(os.path.join(user_directory, source)):

                                valid_sources.append(source)
                        if valid_sources:
                            vector_db.add_sources(valid_sources,user_folder_name,user_directory)
                            context["sources"].extend(valid_sources)
                            #clear_sources_to_add()
                            return jsonify({"success": True, "message": "Successfully added sources"})
                        else:
                            return jsonify({"success": False, "message": "No valid sources provided"})
                else:
                        return jsonify({"success": False, "message": "No sources to add"})


from datetime import datetime  # Add this import statement

@app.route('/insert_email', methods=['POST'])
def insert_email():
    connection = None
    cursor = None

    try:
        # Get the authorization token from the request headers
        token = request.headers.get('Authorization')
        if token and token.startswith('Bearer '):
           actual_token = token.split('Bearer ')[1]
        print(actual_token)

        # Establish database connection
        connection = get_database_connection()
        cursor = connection.cursor()

        # Retrieve user ID using the provided token
        get_user_id_query = 'SELECT user_id, first_name,email FROM Customers WHERE user_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()

        if not user_id_result:
            return jsonify({'error': 'User not found or invalid token'}), 404
        
        user_id = user_id_result[0]

        print(user_id)
        admin_email = user_id_result[2]
        # Extract data from request body
        print(admin_email)
        data = request.json
        name = data.get('name')
        email = data.get('email')
        phone = data.get('phone')
        print(data)
        # Get current timestamp
        current_timestamp = datetime.now().date()

        # Insert or update customer's information into chats_customer table
        insert_query = '''
            INSERT INTO chats_customer
            (user_id, customer_email, customer_username, customer_phone_no, created_at)
            VALUES
            (%s, %s, %s, %s, %s)
        '''
        insert_data = (user_id, email, name, phone, current_timestamp)
        cursor.execute(insert_query, insert_data)
        connection.commit()
        insertemail(admin_email, name, email,phone,user_id,cursor)

        return jsonify({'message': 'Email and phone inserted successfully'}), 200
    
    except mysql.connector.Error as err:
        return jsonify({'error': f"Database error: {err}"}), 500
    
    except Exception as e:
        send_error_email('insert_email', str(e))
        return jsonify({'error': str(e)}), 500
    
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()



from flask import current_app


def insertemail(recipient, name, customeremail, note, user_id, cursor):
    print(user_id)
    print("recipient",recipient)
    smtp_details_query = 'SELECT smtp_server, email, password,port FROM smtp_details WHERE user_id = %s'
    cursor.execute(smtp_details_query, (user_id,))
    smtp_details = cursor.fetchone()
    print(smtp_details)

    if not smtp_details:
        raise Exception("SMTP details not found")

    smtp_server, email_from, password,port = smtp_details
    print(smtp_server)
    with current_app.app_context():
        current_app.config['MAIL_SERVER'] = smtp_server
        current_app.config['MAIL_PORT'] = port
        current_app.config['MAIL_USE_TLS'] = True
     #   current_app.config['MAIL_USE_TLS'] = use_tls
        current_app.config['MAIL_USERNAME'] = email_from
        current_app.config['MAIL_PASSWORD'] =password

        # Initialize mail instance
        mail = Mail(current_app)

    # Configure and send the email
        msg = Message("Customer Email", sender=email_from, recipients=[recipient])
        msg.body = f"Name: {name}\nEmail: {customeremail}\nPhone: {note}"  # Corrected variable name here

        try:
          mail.send(msg)
          print("Email sent successfully!")
        except Exception as e:
           send_error_email('insertemail', str(e))
           print(f"Failed to send email: {str(e)}")

@app.route('/get_customer_info', methods=['GET'])
def get_customer_info():
    connection = None
    cursor = None

    try:
        # Get the authorization token from the request headers
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Authorization header is missing or invalid'}), 401

        # Extract the token from the authorization header
        token = auth_header.split('Bearer ')[1]

        # Establish database connection
        connection = get_database_connection()
        cursor = connection.cursor()

        # Retrieve user ID using the provided token
        get_user_id_query = 'SELECT user_id FROM Customers WHERE api_token = %s'
        cursor.execute(get_user_id_query, (token,))
        user_id_result = cursor.fetchone()

        if not user_id_result:
            return jsonify({'error': 'User not found or invalid token'}), 404

        user_id = user_id_result[0]

        # Retrieve customer information
        get_customer_info_query = '''
            SELECT customer_email, customer_username, customer_phone_no, created_at
            FROM chats_customer
            WHERE user_id = %s
        '''
        cursor.execute(get_customer_info_query, (user_id,))
        customer_info = cursor.fetchall()

        if not customer_info:
           return jsonify({'message': 'Customer information not found'}), 404

# Assuming customer_info is a list of tuples, access the first tuple
        response_data = []

        for customer_data in customer_info:
            customer_dict = {
                    'email': customer_data[0],
                    'name': customer_data[1],
                    'phone': customer_data[2],
                    'created_at': customer_data[3].strftime('%Y-%m-%d')
            }
            response_data.append(customer_dict)

# Prepare response data

        return jsonify(response_data), 200

    except mysql.connector.Error as err:
        return jsonify({'error': f"Database error: {err}"}), 500

    except Exception as e:
        send_error_email('get_customer_info', str(e))
        return jsonify({'error': str(e)}), 500

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


@app.route('/insert', methods=['POST'])
def insert():
    connection = get_database_connection()
    cursor = connection.cursor()
    token = request.headers.get('Authorization')
    print(token)

    try:
        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
            print(actual_token)
        else:
            actual_token = None
            response_data = {"message": "Invalid token format"}
            return jsonify(response_data)

        get_user_id_query = 'SELECT user_id FROM Customers WHERE api_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()
        print(user_id_result)

        if user_id_result:
            user_id = user_id_result[0]
            print(user_id)

            # Extract data from request body
            data = request.json
            name = data.get('name')
            email = data.get('email')

            # Insert or update customer's name and email
            insert_query = '''
              INSERT INTO chats_customer
              (user_id, customer_email, customer_username)
              VALUES
              (%s, %s, %s)
              ON DUPLICATE KEY UPDATE
              customer_email = VALUES(customer_email),
              customer_username = VALUES(customer_username)
            '''
            insert_data = (user_id, email, name)
            cursor.execute(insert_query, insert_data)
            connection.commit()

            return jsonify({'message': 'Email inserted successfully'}), 200
        else:
            return jsonify({'error': 'User ID not provided'}), 400
    except Exception as e:
        send_error_email('insert', str(e))
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        connection.close()


@app.route('/eventurls', methods=['POST'])
def eventurls():
    connection = None
    cursor = None
    try:
        connection = get_database_connection()
        cursor = connection.cursor()
        token = request.headers.get('Authorization')
        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
        else:
            return jsonify({'error': 'Invalid token'}), 401

        get_user_id_query = 'SELECT user_id FROM Customers WHERE api_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()

        if user_id_result:
            user_id = user_id_result[0]
            data = request.json
            urls = data.get('urls', [])

            # Iterate over the urls and perform an insert or update query for each URL
            for url_data in urls:
                insert_query = """
                INSERT INTO events (user_id, url1, url2, url3, description1, description2, description3, title1, title2, title3, image1, image2, image3)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE url1 = VALUES(url1), url2 = VALUES(url2), url3 = VALUES(url3),
                description1 = VALUES(description1), description2 = VALUES(description2), description3 = VALUES(description3),
                title1 = VALUES(title1), title2 = VALUES(title2), title3 = VALUES(title3),
                image1 = VALUES(image1), image2 = VALUES(image2), image3 = VALUES(image3);
                """
                # Extracting data for each URL from url_data dictionary
                url1 = url_data.get('url1')
                url2 = url_data.get('url2')
                url3 = url_data.get('url3')
                description1 = url_data.get('description1')
                description2 = url_data.get('description2')
                description3 = url_data.get('description3')
                title1 = url_data.get('title1')
                title2 = url_data.get('title2')
                title3 = url_data.get('title3')
                image1 = url_data.get('image1')
                image2 = url_data.get('image2')
                image3 = url_data.get('image3')

                # Executing the query with the extracted data
                cursor.execute(insert_query, (user_id, url1, url2, url3, description1, description2, description3, title1, title2, title3, image1, image2, image3))

            connection.commit()  # Commit the transaction after all URLs are processed
            return jsonify({'message': 'URLs saved successfully'}), 200
        else:
            return jsonify({'error': 'User not found'}), 404

    except Exception as e:
        send_error_email('eventurls', str(e))
        connection.rollback()  # Rollback the transaction in case of an error
        return jsonify({'error': str(e)}), 500
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

import base64

@app.route('/get_eventsurl', methods=['GET'])
def get_eventsurl():
    connection = None
    cursor = None
    try:
        connection = get_database_connection()
        cursor = connection.cursor()

        token = request.headers.get('Authorization')
        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
        else:
            return jsonify({'error': 'Invalid token'}), 401

        # Retrieve user ID based on the API token
        get_user_id_query = 'SELECT user_id FROM Customers WHERE api_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()
        if user_id_result:
            user_id = user_id_result[0]

            # Retrieve events for the user
            get_events_query = """
            SELECT url1, url2, url3, description1, description2, description3,
                   title1, title2, title3, image1, image2, image3
            FROM events
            WHERE user_id = %s
            """
            cursor.execute(get_events_query, (user_id,))
            events = cursor.fetchall()

            # Construct JSON response
            event_list = []
            for event in events:
                event_data = {
                    'url1': event[0],
                    'url2': event[1],
                    'url3': event[2],
                    'description1': event[3],
                    'description2': event[4],
                    'description3': event[5],
                    'title1': event[6],
                    'title2': event[7],
                    'title3': event[8],
                }
                # Convert images to base64 strings
                image1_base64 = base64.b64encode(event[9]).decode('utf-8') if event[9] else None
                image2_base64 = base64.b64encode(event[10]).decode('utf-8') if event[10] else None
                image3_base64 = base64.b64encode(event[11]).decode('utf-8') if event[11] else None

                # Add base64 strings to event_data dictionary
                event_data['image1'] = image1_base64
                event_data['image2'] = image2_base64
                event_data['image3'] = image3_base64

                event_list.append(event_data)

            return jsonify({'events': event_list}), 200
        else:
            return jsonify({'error': 'User not found'}), 404

    except Exception as e:
        send_error_email('get_eventsurl', str(e))
        return jsonify({'error': str(e)}), 500
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


def getimage_format(file_name):
    _, extension = os.path.splitext(file_name)
    if extension.lower() == '.jpg' or extension.lower() == '.jpeg':
        return 'jpeg'
    elif extension.lower() == '.png':
        return 'png'
    elif extension.lower() == '.gif':
        return 'gif'
    else:
        return None  # Unknown format

@app.route('/getevent', methods=['GET'])
def getvent():
    cursor = None
    connection = None
    try:
        token = request.headers.get('Authorization')
        print(token)

        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
            connection = get_database_connection()
            cursor = connection.cursor()
            print(actual_token)

            cursor.execute('SELECT user_id FROM Customers WHERE user_token = %s', (actual_token,))
            user_id_result = cursor.fetchone()
            print(user_id_result)
            

            if user_id_result:
                user_id = user_id_result[0]
                print(user_id)

                # Retrieve event URLs
                cursor.execute('SELECT url1, url2, url3, description1, description2, description3, title1, title2, title3 FROM setting_event WHERE user_id = %s', (user_id,))
                urls_result = cursor.fetchone()

                if urls_result:
                    urls = {
                        'url1': urls_result[0],
                        'url2': urls_result[1],
                        'url3': urls_result[2],
                        'description1': urls_result[3],
                        'description2': urls_result[4],
                        'description3': urls_result[5],
                        'title1': urls_result[6],
                        'title2': urls_result[7],
                        'title3': urls_result[8]
                    }

                    # Retrieve directory name
                    cursor.execute('SELECT directory_name FROM Directory WHERE user_id = %s', (user_id,))
                    directory_info = cursor.fetchone()

                    if directory_info:
                        directory_name = directory_info[0]
                        save_folder = os.path.join(BASE_UPLOAD_FOLDER, directory_name, "setting_image")

                        # Get list of all files in the directory
                        files = os.listdir(save_folder)

                        # Filter out non-image files
                        image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]

                        # Prepare a list to store image data
                        image_data = []

                        # Read each image file and append its base64 encoded string to the list
                        for image_file in image_files:
                            with open(os.path.join(save_folder, image_file), 'rb') as file:
                                image_data.append(base64.b64encode(file.read()).decode('utf-8'))

                        # Get the image format for the first image file
                        image_format = getimage_format(image_files[0]) if image_files else None

                        if image_format:
                            response_data = {
                                'urls': urls,
                                'image_data': image_data,
                                'image_format': image_format
                            }
                            return jsonify(response_data), 200
                        else:
                            return jsonify({'error': 'Unknown image format'}), 500
                    else:
                        return jsonify({'error': 'Directory not found for the user'}), 404
                else:
                    return jsonify({'message': 'No URLs found for the user'}), 404
            else:
                return jsonify({'error': 'User not found'}), 404
        else:
            return jsonify({'error': 'Invalid token'}), 401
    except Exception as e:
        send_error_email('getevent', str(e))
        print(f"Error in get_data: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()



ALLOWED_EXT = {'png', 'jpg', 'jpeg', 'gif'}


@app.route('/logo_image', methods=['POST'])
def logo_image():
    try:
        token = request.headers.get('Authorization')
        print("Token:", token)

        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
            print(actual_token)

            # Retrieve user_id and directory name from the database using token
            connection = get_database_connection()
            cursor = connection.cursor()
            cursor.execute('SELECT user_id FROM Customers WHERE api_token = %s', (actual_token,))
            user_id_result = cursor.fetchone()

            if user_id_result:
                user_id = user_id_result[0]
                print("User ID:", user_id)

                cursor.execute('SELECT directory_name FROM Directory WHERE user_id = %s', (user_id,))
                directory_info = cursor.fetchone()
                print(directory_info)

                if directory_info:
                    directory_name = directory_info[0]
                    print("Directory:", directory_name)

                    # Check if the post request has the file part
                    if 'file' not in request.files:
                        return jsonify({'error': 'No file part in the request'}), 400

                    file = request.files['file']

                    # Check if the file is one of the allowed types/extensions
                    if file and allowedfilename(file.filename):

                        save_folder = os.path.join(BASE_UPLOAD_FOLDER, directory_name, "logo")
                        if not os.path.exists(save_folder):
                            os.makedirs(save_folder)

                        # Remove old images if they exist (removed as per your request)

                        # Save the received image file to the specified directory
                        # Use a single name for the image file
                        filename = "logo.jpg"  # You can choose any desired name
                        file.save(os.path.join(save_folder, filename))
                        return jsonify({'message': 'Image uploaded successfully'})
                    else:
                        return jsonify({'error': 'Invalid file type. Allowed file types are: png, jpg, jpeg, gif'}), 400
                else:
                    return jsonify({'error': 'Directory not found for the user'}), 404
            else:
                return jsonify({'error': 'User not found'}), 404
        else:
            return jsonify({'error': 'Invalid token'}), 401
    except Exception as e:
        send_error_email('logo_image', str(e))
        return jsonify({'error': str(e)}), 500
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


def allowedfilename(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

@app.route('/get_logo_image/<image_name>', methods=['GET'])
def get_logo_image(image_name):
    try:
        token = request.headers.get('Authorization')
        print("Token:", token)

        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
            print(actual_token)

            # Retrieve user_id and directory name from the database using token
            connection = get_database_connection()  # Assuming you have a function to get the database connection
            cursor = connection.cursor()
            cursor.execute('SELECT user_id FROM Customers WHERE user_token = %s', (actual_token,))
            user_id_result = cursor.fetchone()

            if user_id_result:
                user_id = user_id_result[0]
                print("User ID:", user_id)

                cursor.execute('SELECT directory_name FROM Directory WHERE user_id = %s', (user_id,))
                directory_info = cursor.fetchone()
                print(directory_info)

                if directory_info:
                    directory_name = directory_info[0]
                    print("Directory:", directory_name)

                    image_directory = os.path.join(BASE_UPLOAD_FOLDER, directory_name, "logo")

                    try:
                        # Attempt to send the image from the directory
                        response = make_response(send_from_directory(image_directory, image_name))
                        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
                        response.headers['Pragma'] = 'no-cache'
                        response.headers['Expires'] = '0'
                        return response
                    except FileNotFoundError:
                        return jsonify({'error': 'Image not found'}), 404
                else:
                    return jsonify({'error': 'Directory not found for the user'}), 404
            else:
                return jsonify({'error': 'User not found'}), 404
        else:
            return jsonify({'error': 'Invalid token'}), 401
    except Exception as e:
        send_error_email('get_logo_image/<image_name>', str(e))
        return jsonify({'error': str(e)}), 500
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

@app.route('/get_logo/<image_name>', methods=['GET'])
def get_logo(image_name):
    try:
        token = request.headers.get('Authorization')
        print("Token:", token)

        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
            print(actual_token)

            # Retrieve user_id and directory name from the database using token
            connection = get_database_connection()  # Assuming you have a function to get the database connection
            cursor = connection.cursor()
            cursor.execute('SELECT user_id FROM Customers WHERE api_token = %s', (actual_token,))
            user_id_result = cursor.fetchone()

            if user_id_result:
                user_id = user_id_result[0]
                print("User ID:", user_id)

                cursor.execute('SELECT directory_name FROM Directory WHERE user_id = %s', (user_id,))
                directory_info = cursor.fetchone()
                print(directory_info)

                if directory_info:
                    directory_name = directory_info[0]
                    print("Directory:", directory_name)

                    image_directory = os.path.join(BASE_UPLOAD_FOLDER, directory_name, "logo")

                    try:
                        # Attempt to send the image from the directory
                        response = make_response(send_from_directory(image_directory, image_name))
                        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
                        response.headers['Pragma'] = 'no-cache'
                        response.headers['Expires'] = '0'
                        return response
                    except FileNotFoundError:
                        return jsonify({'error': 'Image not found'}), 404
                else:
                    return jsonify({'error': 'Directory not found for the user'}), 404
            else:
                return jsonify({'error': 'User not found'}), 404
        else:
            return jsonify({'error': 'Invalid token'}), 401
    except Exception as e:
        send_error_email('get_logo/<image_name>', str(e))
        return jsonify({'error': str(e)}), 500
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


@app.route('/introduction_chatbot_message', methods=['POST'])
def introduction_chatbot_message():
    connection = None
    cursor = None

    try:
        # Get the authorization token from the request headers
        token = request.headers.get('Authorization')
        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]

        # Establish database connection
        connection = get_database_connection()
        cursor = connection.cursor()

        # Retrieve user ID using the provided token
        get_user_id_query = 'SELECT user_id FROM Customers WHERE api_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()

        if not user_id_result:
            return jsonify({'error': 'User not found or invalid token'}), 404

        user_id = user_id_result[0]

        # Extract data from request body
        data = request.json
        introduction = data.get('introduction')
        descriptions = data.get('descriptions')

        # Upsert into introductions table
        upsert_query = '''
            INSERT INTO introductions
            (user_id, introduction, descriptions)
            VALUES
            (%s, %s, %s)
            ON DUPLICATE KEY UPDATE
            introduction = VALUES(introduction),
            descriptions = VALUES(descriptions)
        '''
        upsert_data = (user_id, introduction, descriptions)
        cursor.execute(upsert_query, upsert_data)
        connection.commit()
        
        return jsonify({'message': 'Introduction upserted successfully'}), 200

    except mysql.connector.Error as err:
        return jsonify({'error': f"Database error: {err}"}), 500

    except Exception as e:
        send_error_email('introduction_chatbot_message', str(e))
        return jsonify({'error': str(e)}), 500

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()



@app.route('/introduction_chatbot_message', methods=['GET'])
def get_introduction_chatbot_message():
    connection = None
    cursor = None

    try:
        # Get the authorization token from the request headers
        token = request.headers.get('Authorization')
        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]

        # Establish database connection
        connection = get_database_connection()
        cursor = connection.cursor()

        # Retrieve user ID using the provided token
        get_user_id_query = 'SELECT user_id FROM Customers WHERE user_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()

        if not user_id_result:
            return jsonify({'error': 'User not found or invalid token'}), 404

        user_id = user_id_result[0]

        # Retrieve introduction and descriptions for the user
        get_introduction_query = 'SELECT introduction, descriptions FROM introductions WHERE user_id = %s'
        cursor.execute(get_introduction_query, (user_id,))
        introduction_result = cursor.fetchone()

        if not introduction_result:
            return jsonify({'error': 'Introduction not found for the user'}), 404

        introduction, descriptions = introduction_result

        return jsonify({
            'introduction': introduction,
            'descriptions': descriptions
        }), 200

    except mysql.connector.Error as err:
        return jsonify({'error': f"Database error: {err}"}), 500

    except Exception as e:
        send_error_email('get_introduction_chatbot_message', str(e))
        return jsonify({'error': str(e)}), 500

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()



@app.route('/verify_client_token', methods=['POST'])
def verify_client_token():
    try:
        data = request.json
        token = data.get('client_token')

        if not token:
            return jsonify({'error': 'Missing token'}), 400

        # Establish database connection
        connection = get_database_connection()
        cursor = connection.cursor()

        check_token_query = 'SELECT user_id, first_name, last_name FROM Customers WHERE user_token = %s'
        cursor.execute(check_token_query, (token,))
        user_info = cursor.fetchone()

        if not user_info:
            cursor.close()
            connection.close()
            return jsonify({'error': 'Invalid token'}), 401

        user_id, first_name, last_name = user_info
        username = f"{first_name} {last_name}"
        session['user_id'] = user_id

        cursor.close()
        connection.close()

        user_id = session.get('user_id')
        print(user_id)

        return jsonify({'status': 'success', 'user_id': user_id, 'first_name': first_name, 'last_name': last_name}), 200

    except Exception as e:
        send_error_email('verify_client_token', str(e))
        print(f"Error: {e}")
        return jsonify({'error': 'An error occurred while processing the request'}), 500












@app.route('/get_top_level_components', methods=['GET'])
def get_top_level_components():
    try:
        token = request.headers.get('Authorization')
        print("Token:", token)

        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
            print(actual_token)

            # Retrieve user_id and directory name from the database using token
            connection = get_database_connection()
            cursor = connection.cursor()
            cursor.execute('SELECT user_id FROM Customers WHERE api_token = %s', (actual_token,))
            user_id_result = cursor.fetchone()

            if user_id_result:
                user_id = user_id_result[0]
                print("User ID:", user_id)

                cursor.execute('SELECT directory_name FROM Directory WHERE user_id = %s', (user_id,))
                directory_info = cursor.fetchone()
                print(directory_info)

                if directory_info:
                    directory_name = directory_info[0]
                    print("Directory:", directory_name)

                    # Load JSON file and retrieve top-level components
                    save_file_path = os.path.join(BASE_UPLOAD_FOLDER, directory_name, "setting_pdf", "menu_structure.json")
                    with open(save_file_path, 'r') as file:
                        menu_data = json.load(file)
                        top_level_components = [{'id': key, 'name': value['name']} for key, value in menu_data.items()]

                    return jsonify({'top_level_components': top_level_components})
                else:
                    return jsonify({'error': 'Directory not found for the user'}), 404
            else:
                return jsonify({'error': 'User not found'}), 404
        else:
            return jsonify({'error': 'Invalid token'}), 401
    except Exception as e:
        send_error_email('get_top_level_components', str(e))
        return jsonify({'error': str(e)}), 500
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


@app.route('/submit_email', methods=['POST'])
def submit_email():
    connection = None
    cursor = None

    try:
        # Get the authorization token from the request headers
        token = request.headers.get('Authorization')
        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
        print(actual_token)

        # Establish database connection
        connection = get_database_connection()
        cursor = connection.cursor()

        # Retrieve user ID using the provided token
        get_user_id_query = 'SELECT user_id, first_name, email FROM Customers WHERE user_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()

        if not user_id_result:
            return jsonify({'error': 'User not found or invalid token'}), 404

        user_id = user_id_result[0]
        admin_email = user_id_result[2]  # Index 2 corresponds to the email field

        print(user_id)
        print("admin=", admin_email)
        # Extract data from request body
        data = request.json
        name = data.get('name')
        customeremail = data.get('email')
        note = data.get('note')
        print(name)

        # Get current timestamp
        current_timestamp = datetime.now().date()
        sendemails(admin_email, name, customeremail, note, user_id)
        print("call_function")

        # Insert data into the database
        insert_query = '''
            INSERT INTO chats_customer (user_id, customer_email, customer_username, customer_phone_no, created_at)
            VALUES (%s, %s, %s, %s, %s)
        '''
        insert_data = (user_id, customeremail, name, note, current_timestamp)
        cursor.execute(insert_query, insert_data)
        connection.commit()

        return jsonify({'message': 'Email and phone inserted successfully'}), 200

    except mysql.connector.Error as err:
        return jsonify({'error': f"Database error: {err}"}), 500

    except Exception as e:
        send_error_email('submit_email', str(e))
        return jsonify({'error': str(e)}), 500

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


def sendemails(recipient, name, customeremail, note, user_id):
    connection = get_database_connection()
    cursor = connection.cursor()
    print("inside func=", user_id)
    print(user_id)
    print("recipient", recipient)
    smtp_details_query = 'SELECT smtp_server, email, password, port FROM smtp_details WHERE user_id = %s'
    cursor.execute(smtp_details_query, (user_id,))
    smtp_details = cursor.fetchone()
    print(smtp_details)

    if not smtp_details:
        raise Exception("SMTP details not found")

    smtp_server, email_from, password, port = smtp_details
    print(smtp_server)
    
    with current_app.app_context():
        current_app.config['MAIL_SERVER'] = smtp_server
        current_app.config['MAIL_PORT'] = port
        current_app.config['MAIL_USE_TLS'] = True
        current_app.config['MAIL_USERNAME'] = email_from
        current_app.config['MAIL_PASSWORD'] = password

        # Initialize mail instance
        mail = Mail(current_app)

        # Configure and send the email
        subject = "Customer Email"
        msg = Message(subject, sender=email_from, recipients=[recipient])
        msg.body = f"Name: {name}\nEmail: {customeremail}\nNote: {note}"

        try:
            mail.send(msg)
            print("Email sent successfully!")
        except Exception as e:
            print(f"Failed to send email: {str(e)}")






from flask import request

import base64

from flask import send_from_directory

from flask import jsonify

@app.route('/get_pdf_settings', methods=['GET'])
def get_pdf_settings():
    connection = get_database_connection()
    cursor = connection.cursor()
    try:
        token = request.headers.get('Authorization')
        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
        else:
            return jsonify({'error': 'Invalid token'}), 401

        get_user_id_query = 'SELECT user_id FROM Customers WHERE api_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()

        if user_id_result:
            user_id = user_id_result[0]
            cursor.execute('SELECT * FROM PDF_settings WHERE user_id = %s', (user_id,))
            pdf_settings = cursor.fetchone()
            print(pdf_settings)

            if pdf_settings:
                # Construct PDF settings data
                pdf_data = {
                    'title1': pdf_settings[5],
                    'title2': pdf_settings[6],
                    'title3': pdf_settings[7],
                    'description1': pdf_settings[8],
                    'description2': pdf_settings[9],
                    'description3': pdf_settings[10]
                }
                return jsonify(pdf_data)
            else:
                return jsonify({'error': 'PDF settings not found'}), 404
        else:
            return jsonify({'error': 'User not found'}), 404

    except Exception as e:
        send_error_email('get_pdf_settings', str(e))
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        connection.close()




@app.route('/pdf_settings', methods=['POST'])
def upsert_pdf_settings():
    connection = get_database_connection()
    cursor = connection.cursor()
    try:
        # Check if all required form fields and files exist
        if all(field in request.form for field in ['title1', 'title2', 'title3', 'description1', 'description2', 'description3']) and \
           all(field in request.files for field in ['pdf_file1', 'pdf_file2', 'pdf_file3']):

            # Extract data from form fields
            title1 = request.form['title1']
            title2 = request.form['title2']
            title3 = request.form['title3']
            description1 = request.form['description1']
            description2 = request.form['description2']
            description3 = request.form['description3']

            # Extract file objects
            pdf_file1 = request.files['pdf_file1']
            pdf_file2 = request.files['pdf_file2']
            pdf_file3 = request.files['pdf_file3']
            print(pdf_file1)
            token = request.headers.get('Authorization')
            if token and token.startswith('Bearer '):
                actual_token = token.split('Bearer ')[1]
            else:
                return jsonify({'error': 'Invalid token'}), 401

            get_user_id_query = 'SELECT user_id FROM Customers WHERE api_token = %s'
            cursor.execute(get_user_id_query, (actual_token,))
            user_id_result = cursor.fetchone()

            if user_id_result:
                user_id = user_id_result[0]
                print("pdf=",user_id)
                cursor.execute('SELECT directory_name FROM Directory WHERE user_id = %s', (user_id,))
                directory_info = cursor.fetchone()
                print("dir=",directory_info)

                directory_name = directory_info[0]
                print("Directory:", directory_name)
                # Save uploaded files to the correct directory
                for pdf_file, title in [(pdf_file1, title1), (pdf_file2, title2), (pdf_file3, title3)]:
                    if pdf_file:
                        print(pdf_file)
                        filename = secure_filename(pdf_file.filename)
                        print(filename)
                        pdf_primary_directory=os.path.join(BASE_UPLOAD_FOLDER, directory_name,"pdfs")

                        pdf_directory = os.path.join(pdf_primary_directory,filename)
                        os.makedirs(pdf_primary_directory, exist_ok=True)
                        os.makedirs(pdf_directory, exist_ok=True)

                        pdf_file.save(os.path.join(pdf_directory, filename))

                # Execute upsert query

                upsert_query = """
                INSERT INTO PDF_settings
               (user_id, pdf_file1, pdf_file2, pdf_file3,
                title1, title2, title3,
                 description1, description2, description3)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                pdf_file1 = IF(VALUES(pdf_file1) IS NOT NULL, VALUES(pdf_file1), pdf_file1),
                pdf_file2 = IF(VALUES(pdf_file2) IS NOT NULL, VALUES(pdf_file2), pdf_file2),
                pdf_file3 = IF(VALUES(pdf_file3) IS NOT NULL, VALUES(pdf_file3), pdf_file3),
                title1 = IF(VALUES(title1) IS NOT NULL, VALUES(title1), title1),
                title2 = IF(VALUES(title2) IS NOT NULL, VALUES(title2), title2),
                title3 = IF(VALUES(title3) IS NOT NULL, VALUES(title3), title3),
                description1 = IF(VALUES(description1) IS NOT NULL, VALUES(description1), description1),
                description2 = IF(VALUES(description2) IS NOT NULL, VALUES(description2), description2),
                description3 = IF(VALUES(description3) IS NOT NULL, VALUES(description3), description3)
            """
                cursor.execute(upsert_query, (
                    user_id, pdf_file1.filename if pdf_file1 else None,
                    pdf_file2.filename if pdf_file2 else None,
                    pdf_file3.filename if pdf_file3 else None,
                    title1, title2, title3,
                    description1, description2, description3
                ))
                connection.commit()  # Commit the transaction
                return jsonify({'message': 'PDF settings upserted successfully'})

            else:
                return jsonify({'error': 'User not found'}), 404
        else:
            return jsonify({'error': 'Missing form fields or files'}), 400

    except Exception as e:
        send_error_email('pdf_settings', str(e))
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        connection.close()



@app.route('/pdf_chatbot_setting_delete', methods=['POST'])
def pdf_chatbot_setting_delete():
    try:
        token = request.headers.get('Authorization')
        print("Token:", token)

        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
            print(actual_token)

            # Retrieve user_id and directory name from the database using token
            connection = get_database_connection()
            cursor = connection.cursor()
            cursor.execute('SELECT user_id FROM Customers WHERE api_token = %s', (actual_token,))
            user_id_result = cursor.fetchone()
            if user_id_result:
                user_id = user_id_result[0]
                print("User ID:", user_id)

                cursor.execute('SELECT directory_name FROM Directory WHERE user_id = %s', (user_id,))
                directory_info = cursor.fetchone()
                if directory_info:
                    directory_name = directory_info[0]
                    print("Directory:", directory_name)

                    # Get the filename to delete
                    data = request.json
                    print("Request Data:", data)
                    if 'filenames' in data:
                        filenames = data['filenames']
                        print("Filenames:", filenames[0])
                        for filename in filenames:
                            
                            file_path = os.path.join(BASE_UPLOAD_FOLDER, directory_name, "chatbot_pdf_setting",filename)

                            # Check if file exists and delete it
                            if os.path.exists(file_path):
                                os.remove(file_path)
                            print(file_path)
                            # Remove entry from the database
                            cursor.execute('DELETE FROM documents WHERE user_id = %s AND file_name = %s', (user_id, filename))

                        # Commit the database changes
                        connection.commit()

                        return jsonify({'message': 'Files deleted successfully'}), 200
                    else:
                        return jsonify({'error': 'Filename not provided in request'}), 400
                else:
                    return jsonify({'error': 'Directory not found'}), 404
            else:
                return jsonify({'error': 'User not found'}), 404
        else:
            return jsonify({'error': 'Invalid token'}), 401
    except Exception as e:
        send_error_email('pdf_chatbot_setting_delete', str(e))
        return jsonify({'error': str(e)}), 500
    finally:
        if cursor:
            cursor.close()

@app.route('/image_setting_delete', methods=['POST'])
def image_setting_delete():
    try:
        token = request.headers.get('Authorization')
        print("Token:", token)

        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
            print(actual_token)

            # Retrieve user_id and directory name from the database using token
            connection = get_database_connection()
            cursor = connection.cursor()
            cursor.execute('SELECT user_id FROM Customers WHERE api_token = %s', (actual_token,))
            user_id_result = cursor.fetchone()
            if user_id_result:
                user_id = user_id_result[0]
                print("User ID:", user_id)

                cursor.execute('SELECT directory_name FROM Directory WHERE user_id = %s', (user_id,))
                directory_info = cursor.fetchone()
                if directory_info:
                    directory_name = directory_info[0]
                    print("Directory:", directory_name)

                    # Get the filename to delete
                    data = request.json
                    print("Request Data:", data)
                    if 'filenames' in data:
                        filenames = data['filenames']
                        print("Filenames:", filenames[0])
                        for filename in filenames:

                            file_path = os.path.join(BASE_UPLOAD_FOLDER, directory_name, "chatbot_image_setting",filename)

                            # Check if file exists and delete it
                            if os.path.exists(file_path):
                                os.remove(file_path)
                            print(file_path)
                            # Remove entry from the database
                            cursor.execute('DELETE FROM events_data WHERE user_id = %s AND file_name = %s', (user_id, filename))

                        # Commit the database changes
                        connection.commit()

                        return jsonify({'message': 'Files deleted successfully'}), 200
                    else:
                        return jsonify({'error': 'Filename not provided in request'}), 400
                else:
                    return jsonify({'error': 'Directory not found'}), 404
            else:
                return jsonify({'error': 'User not found'}), 404
        else:
            return jsonify({'error': 'Invalid token'}), 401
    except Exception as e:
        send_error_email('image_setting_delete', str(e))
        return jsonify({'error': str(e)}), 500
    finally:
        if cursor:
            cursor.close()




ALLOWED = {'pdf'}


# Function to check if a filename has an allowed extension
def allowed_pdf(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED

# Route for uploading PDF files
@app.route('/upload_pdf_chatbot_setting', methods=['POST'])
def upload_pdf_chatbot_setting():
    connection = None
    cursor = None

    try:
        token = request.headers.get('Authorization')

        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
            print(actual_token)

            # Retrieve user_id from database using token
            connection = get_database_connection()
            cursor = connection.cursor()
            cursor.execute('SELECT user_id FROM Customers WHERE api_token = %s', (actual_token,))
            user_id_result = cursor.fetchone()

            if user_id_result:
                user_id = user_id_result[0]
                print(user_id)
                # Check if PDF file is provided in the request
                if 'pdf' not in request.files:
                    return jsonify({'error': 'No PDF file provided'}), 400

                file = request.files['pdf']

                if file.filename == '':
                    return jsonify({'error': 'No selected file'}), 400

                if file and allowed_pdf(file.filename):
                    filename = secure_filename(file.filename)
                    print(filename)

                    # Generate unique ID for the PDF
                    pdf_id = request.form.get('pdf_id')  # Generating UUID
                    print(pdf_id)
                    # Retrieve title and description from the request
                    title = request.form.get('title')
                    print(title)
                    description = request.form.get('description')
                    print(description)

                    # Retrieve directory name from the database based on user ID
                    cursor.execute('SELECT directory_name FROM Directory WHERE user_id = %s', (user_id,))
                    directory_info = cursor.fetchone()

                    if directory_info:
                        directory_name = directory_info[0]
                        user_directory_path = os.path.join(BASE_UPLOAD_FOLDER, directory_name, "chatbot_pdf_setting")

                        # Create directory if it doesn't exist
                        if not os.path.exists(user_directory_path):
                            os.makedirs(user_directory_path)
                        print(user_directory_path)
                        file.save(os.path.join(user_directory_path, filename))

                        # Update database with the file path, PDF ID, and other information
                        cursor.execute("""
INSERT INTO documents (pdf_id, user_id, file_name, title, description)
VALUES (%s, %s, %s, %s, %s)
ON DUPLICATE KEY UPDATE
    file_name = IF(pdf_id = VALUES(pdf_id) AND user_id = VALUES(user_id), VALUES(file_name), file_name),
    title = IF(pdf_id = VALUES(pdf_id) AND user_id = VALUES(user_id), VALUES(title), title),
    description = IF(pdf_id = VALUES(pdf_id) AND user_id = VALUES(user_id), VALUES(description), description);

                                       """, (pdf_id, user_id, filename, title, description))

                        connection.commit()

                        return jsonify({'message': 'PDF file uploaded successfully', 'pdf_id': pdf_id}), 200
                    else:
                        return jsonify({'error': 'Directory not found for the user'}), 404
                else:
                    return jsonify({'error': 'Invalid file format, only PDF files allowed'}), 400
            else:
                return jsonify({'error': 'User not found'}), 404
        else:
            return jsonify({'error': 'Invalid token'}), 401
    except Exception as e:
        send_error_email('upload_pdf_chatbot_setting', str(e))
        return jsonify({'error': str(e)}), 500
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()














ALLOWEDIMG = {'jpg', 'jpeg', 'png', 'gif'}

# Function to check if a filename has an allowed extension
def allowed_image(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWEDIMG

# Route for uploading image files
@app.route('/upload_image_chatbot_setting', methods=['POST'])
def upload_image_chatbot_setting():
    connection = None
    cursor = None

    try:
        token = request.headers.get('Authorization')

        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
            print(actual_token)

            # Retrieve user_id from database using token
            connection = get_database_connection()
            cursor = connection.cursor()
            cursor.execute('SELECT user_id FROM Customers WHERE api_token = %s', (actual_token,))
            user_id_result = cursor.fetchone()

            if user_id_result:
                user_id = user_id_result[0]
                print(user_id)
                # Check if image file is provided in the request
                if 'pdf' not in request.files:
                    return jsonify({'error': 'No image file provided'}), 400

                file = request.files['pdf']
                print(file)
                if file.filename == '':
                    return jsonify({'error': 'No selected file'}), 400

                if file and allowed_image(file.filename):
                    filename = secure_filename(file.filename)
                    print(filename)
       
                    # Generate unique ID for the image
                    image_id = request.form.get('pdf_id')  # Generating UUID
                    print(image_id)
                    # Retrieve title, description, and URL from the request
                    title = request.form.get('title')
                    print(title)
                    description = request.form.get('description')
                    print(description)
                    url = request.form.get('url')
                    print(url)

                    # Retrieve directory name from the database based on user ID
                    cursor.execute('SELECT directory_name FROM Directory WHERE user_id = %s', (user_id,))
                    directory_info = cursor.fetchone()


                    if directory_info:
                        directory_name = directory_info[0]
                        user_directory_path = os.path.join(BASE_UPLOAD_FOLDER, directory_name, "chatbot_image_setting")
                        print(directory_name)

                        # Create directory if it doesn't exist
                        if not os.path.exists(user_directory_path):
                            os.makedirs(user_directory_path)
                        print(user_directory_path)
                        file.save(os.path.join(user_directory_path, filename))

                        # Update database with the file path, image ID, URL, and other information
                        cursor.execute("""
                            INSERT INTO events_data (image_id, user_id, file_name, title, description, url)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            ON DUPLICATE KEY UPDATE
                                file_name = IF(image_id = VALUES(image_id) AND user_id = VALUES(user_id), VALUES(file_name), file_name),
                                title = IF(image_id = VALUES(image_id) AND user_id = VALUES(user_id), VALUES(title), title),
                                description = IF(image_id = VALUES(image_id) AND user_id = VALUES(user_id), VALUES(description), description),
                                url = IF(image_id = VALUES(image_id) AND user_id = VALUES(user_id), VALUES(url), url);
                        """, (image_id, user_id, filename, title, description, url))

                        connection.commit()

                        return jsonify({'message': 'Image file uploaded successfully', 'image_id': image_id}), 200
                    else:
                        return jsonify({'error': 'Directory not found for the user'}), 404
                else:
                    return jsonify({'error': 'Invalid file format, only image files allowed'}), 400
            else:
                return jsonify({'error': 'User not found'}), 404
        else:
            return jsonify({'error': 'Invalid token'}), 401
    except Exception as e:
        send_error_email('upload_image_chatbot_setting', str(e))
        return jsonify({'error': str(e)}), 500
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()













@app.route('/get_image_chatbot_setting', methods=['GET'])
def get_image_chatbot_setting():
    connection = None
    cursor = None

    try:
        token = request.headers.get('Authorization')

        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]

            # Retrieve user_id from database using token
            connection = get_database_connection()
            cursor = connection.cursor()
            cursor.execute('SELECT user_id FROM Customers WHERE api_token = %s', (actual_token,))
            user_id_result = cursor.fetchone()

            if user_id_result:
                user_id = user_id_result[0]
                print(user_id)
                # Retrieve PDF files uploaded by the user
                cursor.execute("""
                    SELECT file_name, title, description,url
                    FROM  events_data
                    WHERE user_id = %s
                """, (user_id,))

                user_pdfs = cursor.fetchall()
                print(user_pdfs)
                if user_pdfs:
                    pdf_list = []
                    print(pdf_list)
                    for pdf_info in user_pdfs:
                        print(pdf_info)
                        file_name, title, description,url = pdf_info
                        pdf_list.append({
                            'file_name': pdf_info[0],
                            'title': pdf_info[1],
                            'description': pdf_info[2],
                            'url': pdf_info[3]
                        })
                    return jsonify({'pdfs': pdf_list}), 200
                else:
                    return jsonify({'message': 'No PDFs found for the user'}), 404
            else:
                return jsonify({'error': 'User not found'}), 404
        else:
            return jsonify({'error': 'Invalid token'}), 401
    except Exception as e:
        send_error_email('get_image_chatbot_setting', str(e))
        return jsonify({'error': str(e)}), 500
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()




@app.route('/get_pdf_chatbot_setting', methods=['GET'])
def get_pdf_chatbot_setting():
    connection = None
    cursor = None

    try:
        token = request.headers.get('Authorization')

        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]

            # Retrieve user_id from database using token
            connection = get_database_connection()
            cursor = connection.cursor()
            cursor.execute('SELECT user_id FROM Customers WHERE api_token = %s', (actual_token,))
            user_id_result = cursor.fetchone()

            if user_id_result:
                user_id = user_id_result[0]
                print(user_id)
                # Retrieve PDF files uploaded by the user
                cursor.execute("""
                    SELECT file_name, title, description
                    FROM documents
                    WHERE user_id = %s
                """, (user_id,))

                user_pdfs = cursor.fetchall()
                print(user_pdfs)
                if user_pdfs:
                    pdf_list = []
                    print(pdf_list)
                    for pdf_info in user_pdfs:
                        print(pdf_info)
                        file_name, title, description = pdf_info
                        pdf_list.append({
                            'file_name': pdf_info[0],
                            'title': pdf_info[1],
                            'description': pdf_info[2]
                        })
                    return jsonify({'pdfs': pdf_list}), 200
                else:
                    return jsonify({'message': 'No PDFs found for the user'}), 404
            else:
                return jsonify({'error': 'User not found'}), 404
        else:
            return jsonify({'error': 'Invalid token'}), 401
    except Exception as e:
        send_error_email('get_pdf_chatbot_setting', str(e))
        return jsonify({'error': str(e)}), 500
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


@app.route('/events_settings', methods=['POST'])
def events_settings():
    connection = get_database_connection()
    cursor = connection.cursor()
    try:
        # Check if all required form fields, files, and URLs exist
        if all(field in request.form for field in ['title1', 'title2', 'title3', 'description1', 'description2', 'description3']) and \
           all(field in request.files for field in ['image_file1', 'image_file2', 'image_file3']) and \
           all(field in request.form for field in ['url1', 'url2', 'url3']):

            # Extract data from form fields
            title1 = request.form['title1']
            title2 = request.form['title2']
            title3 = request.form['title3']
            description1 = request.form['description1']
            description2 = request.form['description2']
            description3 = request.form['description3']

            # Extract file objects
            image_file1 = request.files['image_file1']
            image_file2 = request.files['image_file2']
            image_file3 = request.files['image_file3']

            # Extract URLs
            url1 = request.form.get('url1', '')
            url2 = request.form.get('url2', '')
            url3 = request.form.get('url3', '')

            token = request.headers.get('Authorization')
            if token and token.startswith('Bearer '):
                actual_token = token.split('Bearer ')[1]
            else:
                return jsonify({'error': 'Invalid token'}), 401

            get_user_id_query = 'SELECT user_id FROM Customers WHERE api_token = %s'
            cursor.execute(get_user_id_query, (actual_token,))
            user_id_result = cursor.fetchone()

            if user_id_result:
                user_id = user_id_result[0]

                cursor.execute('SELECT user_directory FROM user_directory WHERE user_id = %s', (user_id,))
                directory_info = cursor.fetchone()

                if directory_info:
                    user_directory = directory_info[0]
                    print(user_directory)

                    # Save uploaded files to the correct directory
                    for image_file, title, url in [(image_file1, title1, url1), (image_file2, title2, url2), (image_file3, title3, url3)]:
                        if image_file:
                            filename = secure_filename(image_file.filename)
                            image_primary_directory = os.path.join(BASE_UPLOAD_FOLDER, user_directory, "event_images")
                            print(image_primary_directory)
                            os.makedirs(image_primary_directory, exist_ok=True)
                            image_file_path = os.path.join(image_primary_directory, filename)
                            image_file.save(image_file_path)

                    # Execute upsert query
                    upsert_query = """
                        INSERT INTO img_data
                        (user_id, img1, img2, img3,
                         description1, description2, description3,
                         url1, url2, url3)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                        img1 = IF(VALUES(img1) IS NOT NULL, VALUES(img1), img1),
                        img2 = IF(VALUES(img2) IS NOT NULL, VALUES(img2), img2),
                        img3 = IF(VALUES(img3) IS NOT NULL, VALUES(img3), img3),
                        description1 = IF(VALUES(description1) IS NOT NULL, VALUES(description1), description1),
                        description2 = IF(VALUES(description2) IS NOT NULL, VALUES(description2), description2),
                        description3 = IF(VALUES(description3) IS NOT NULL, VALUES(description3), description3),
                        url1 = IF(VALUES(url1) IS NOT NULL, VALUES(url1), url1),
                        url2 = IF(VALUES(url2) IS NOT NULL, VALUES(url2), url2),
                        url3 = IF(VALUES(url3) IS NOT NULL, VALUES(url3), url3)
                    """
                    cursor.execute(upsert_query, (
                        user_id, image_file1.filename if image_file1 else None,
                        image_file2.filename if image_file2 else None,
                        image_file3.filename if image_file3 else None,
                        description1, description2, description3,
                        url1, url2, url3
                    ))
                    connection.commit()  # Commit the transaction
                    return jsonify({'message': 'PDF settings upserted successfully'})

                else:
                    return jsonify({'error': 'User directory not found'}), 404

            else:
                return jsonify({'error': 'User not found'}), 404

        else:
            return jsonify({'error': 'Missing form fields or files'}), 400

    except Exception as e:
        send_error_email('events_settings', str(e))
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        connection.close()




import os
from flask import send_from_directory, jsonify, make_response

@app.route('/pdfsettings_get', methods=['GET'])
def pdfsettings_get():
    connection = get_database_connection()
    cursor = connection.cursor()
    try:
        token = request.headers.get('Authorization')
        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
        else:
            return jsonify({'error': 'Invalid token'}), 401

        get_user_id_query = 'SELECT user_id FROM Customers WHERE user_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()

        if user_id_result:
            user_id = user_id_result[0]
            cursor.execute('SELECT directory_name FROM Directory WHERE user_id = %s', (user_id,))
            directory_info = cursor.fetchone()
            directory_name = directory_info[0]

            cursor.execute('SELECT pdf_id,file_name, title, description FROM documents WHERE user_id = %s', (user_id,))
            pdf_settings = cursor.fetchall()

            if pdf_settings:
                pdf_data = []
                for pdf_info in pdf_settings:
                    pdf_id, file_name, title, description = pdf_info
                    file_path = os.path.join(BASE_UPLOAD_FOLDER, directory_name, "chatbot_pdf_setting", file_name)
                    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                        pdf_data.append({
                            'pdf_id': pdf_id,
                            'file_name': file_name,
                            'title': title,
                            'description': description,
                            'file_path': file_path
                        })

                return jsonify({'pdf_data': pdf_data})
            else:
                return jsonify({'error': 'PDF settings not found'}), 404
        else:
            return jsonify({'error': 'User not found'}), 404

    except Exception as e:
        send_error_email('pdfsettings_get', str(e))
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        connection.close()


import base64

@app.route('/eventsettings_get', methods=['GET'])
def eventsettings_get():
    connection = get_database_connection()
    cursor = connection.cursor()
    try:
        token = request.headers.get('Authorization')
        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
        else:
            return jsonify({'error': 'Invalid token'}), 401

        get_user_id_query = 'SELECT user_id FROM Customers WHERE user_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()

        if user_id_result:
            user_id = user_id_result[0]
            cursor.execute('SELECT directory_name FROM Directory WHERE user_id = %s', (user_id,))
            directory_info = cursor.fetchone()
            directory_name = directory_info[0]

            cursor.execute('SELECT image_id, file_name, title, description, url FROM events_data WHERE user_id = %s', (user_id,))
            pdf_settings = cursor.fetchall()

            if pdf_settings:
                event_data = []
                for pdf_info in pdf_settings:
                    pdf_id, file_name, title, description, url = pdf_info
                    file_path = os.path.join(BASE_UPLOAD_FOLDER, directory_name, "chatbot_image_setting", file_name)
                    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                        with open(file_path, "rb") as image_file:
                            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                        event_data.append({
                            'image_id': pdf_id,
                            'file_name': file_name,
                            'title': title,
                            'description': description,
                            'image': encoded_image,
                            'url': url
                        })

                return jsonify({'event_data': event_data})
            else:
                return jsonify({'error': 'PDF settings not found'}), 404
        else:
            return jsonify({'error': 'User not found'}), 404

    except Exception as e:
        send_error_email('eventsettings_get', str(e))
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        connection.close()




@app.route('/pdfdownload/<filename>', methods=['GET'])
def downloadpdf(filename):
    try:
        token = request.headers.get('Authorization')
        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
        else:
            return jsonify({'error': 'Invalid token'}), 401

        connection = get_database_connection()
        cursor = connection.cursor()

        get_user_id_query = 'SELECT user_id FROM Customers WHERE user_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()

        if user_id_result:
            user_id = user_id_result[0]
            cursor.execute('SELECT directory_name FROM Directory WHERE user_id = %s', (user_id,))
            directory_info = cursor.fetchone()

            if directory_info:
                directory_name = directory_info[0]
                print(directory_name)
                pdf_directory = os.path.join(BASE_UPLOAD_FOLDER, directory_name, "chatbot_pdf_setting")

                # Check if the directory exists
                print(pdf_directory)
                if os.path.exists(pdf_directory):
                    # Debugging: Print the directory path
                    print("PDF Directory:", pdf_directory)

                    # Traverse the directory to find the PDF file
                    for root, _, files in os.walk(pdf_directory):
                        for file in files:
                            if file.lower() == filename.lower() and file.endswith('.pdf'):
                                file_path = os.path.join(root, file)
                                # Debugging: Print the file path
                                print("File Path:", file_path)
                                # Serve the file
                                return send_from_directory(root,file, as_attachment=True)

                    # If file is not found
                    return jsonify({'error': 'File not found'}), 404
                else:
                    return jsonify({'error': 'Directory not found'}), 404
            else:
                return jsonify({'error': 'Directory information not found'}), 404
        else:
            return jsonify({'error': 'User not found'}), 404

    except Exception as e:
        send_error_email('pdfdownload/<filename>', str(e))
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        connection.close()



@app.route('/get_top_level_menu_items', methods=['GET'])
def get_top_level_menu_items():
    try:
        token = request.headers.get('Authorization')
        print("Token:", token)

        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
            print(actual_token)

            # Retrieve user_id and directory name from the database using token
            connection = get_database_connection()
            cursor = connection.cursor()
            cursor.execute('SELECT user_id FROM Customers WHERE api_token = %s', (actual_token,))
            user_id_result = cursor.fetchone()

            if user_id_result:
                user_id = user_id_result[0]
                print("User ID:", user_id)

                cursor.execute('SELECT directory_name FROM Directory WHERE user_id = %s', (user_id,))
                directory_info = cursor.fetchone()
                print(directory_info)

                if directory_info:
                    directory_name = directory_info[0]
                    print("Directory:", directory_name)

                    # Load JSON file and retrieve top level menu items
                    save_file_path = os.path.join(BASE_UPLOAD_FOLDER, directory_name, "setting_pdf", "menu_structure.json")
                    with open(save_file_path, 'r') as file:
                        menu_data = json.load(file)
                        top_level_menu_items = [{'id': item['MenuId'], 'name': item['Menu'], 'type': item.get('Type', '')} for item in menu_data]
                        return jsonify(top_level_menu_items)
                else:
                    return jsonify([])  # Return an empty list if directory not found for the user
            else:
                return jsonify([])  # Return an empty list if user not found
        else:
            return jsonify([]), 401  # Return an empty list with status code 401 for invalid token
    except Exception as e:
        send_error_email('get_top_level_menu_items', str(e))
        return jsonify([]), 500  # Return an empty list with status code 500 for any exception
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


@app.route('/get_submenu_items/<top_level_id>', methods=['GET'])
def get_submenu_items(top_level_id):
    try:

        token = request.headers.get('Authorization')
        print("Token:", token)

        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
            print(actual_token)

            # Retrieve user_id and directory name from the database using token
            connection = get_database_connection()
            cursor = connection.cursor()
            cursor.execute('SELECT user_id FROM Customers WHERE api_token = %s', (actual_token,))
            user_id_result = cursor.fetchone()

            if user_id_result:
                user_id = user_id_result[0]
                print("User ID:", user_id)

                cursor.execute('SELECT directory_name FROM Directory WHERE user_id = %s', (user_id,))
                directory_info = cursor.fetchone()
                print(directory_info)

                if directory_info:
                    directory_name = directory_info[0]
                    print("Directory:", directory_name)

                    # Load JSON file and retrieve submenu items for the given top level id
                    save_file_path = os.path.join(BASE_UPLOAD_FOLDER, directory_name, "setting_pdf", "menu_structure.json")
                    with open(save_file_path, 'r') as file:
                        menu_data = json.load(file)
                        for menu_item in menu_data:
                            if menu_item['MenuId'] == top_level_id:
                                submenu_items = menu_item.get('Submenu', [])
                                submenu_components = [{'id': item['SubmenuId'], 'name': item['Submenu'], 'type': item.get('Type', '')} for item in submenu_items]
                                return jsonify(submenu_components)
                        return jsonify([])  # Return an empty list if top level id not found
                else:
                    return jsonify([])  # Return an empty list if directory not found for the user
            else:
                return jsonify([])  # Return an empty list if user not found
        else:
            return jsonify([]), 401  # Return an empty list with status code 401 for invalid token
    except Exception as e:
        send_error_email('get_submenu_items/<top_level_id>', str(e))
        return jsonify([]), 500  # Return an empty list with status code 500 for any exception
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()





@app.route("/upload_file_service1", methods=['POST'])
def upload_file_service1():
    response_data = {"success": False, "message": ""}
    try:
        file = request.files.get('file')
        if file:
            # Check if file is allowed
            if not allowed_file(file.filename):
                response_data["message"] = "Only PDF files are allowed"
                return jsonify(response_data)

            # Check file size
            if get_file_size(file) > MAX_FILE_SIZE:
                response_data["message"] = "File size exceeds maximum limit (10 MB)"
                return jsonify(response_data)

            # Initialize cursor after establishing connection
            connection = get_database_connection()
            cursor = connection.cursor()

            token = request.headers.get('Authorization')

            # Check if the token starts with 'Bearer ' and extract the actual token
            if token and token.startswith('Bearer '):
                actual_token = token.split('Bearer ')[1]
            else:
                # Handle the case when the token is not in the expected format
                actual_token = None
                response_data["message"] = "Invalid token format"
                cursor.close()
                connection.close()
                return jsonify(response_data)

            # Retrieve the user's ID from the Customers table based on the token
            get_user_id_query = 'SELECT user_id, first_name FROM Customers WHERE api_token = %s'
            cursor.execute(get_user_id_query, (actual_token,))
            user_id_result = cursor.fetchone()

            if user_id_result:
                user_id, first_name = user_id_result
                user_folder_name = f"{first_name}_{user_id}"

                # Retrieve the user's directory name from the Directory table
                get_directory_query = 'SELECT directory_name FROM Directory WHERE user_id = %s'
                cursor.execute(get_directory_query, (user_id,))
                directory_info = cursor.fetchone()

                if directory_info:
                    directory_name = directory_info[0]
                    user_directory_path = os.path.join(BASE_UPLOAD_FOLDER, directory_name, "service1")
                    os.makedirs(user_directory_path, exist_ok=True)

                    filename = secure_filename(file.filename)
                    path = os.path.join(user_directory_path, filename)
                    file.save(path)
                    insert_filename_query = 'INSERT INTO FileLogs (user_id, file_name) VALUES (%s, %s) ON DUPLICATE KEY UPDATE user_id = VALUES(user_id), file_name = VALUES(file_name)'

                    cursor.execute(insert_filename_query, (user_id, filename))
                    connection.commit()  # Commit the transaction

                    response_data["success"] = True
                    response_data["message"] = "File uploaded successfully"
                else:
                    response_data["message"] = "User directory not found"
            else:
                response_data["message"] = "Invalid token"

            cursor.close()
            connection.close()

        else:
            response_data["message"] = "No file provided"

    except Exception as e:
        send_error_email('upload_file_service1', str(e))
        print(f"Error: {e}")
        response_data["message"] = "An error occurred while processing the request"

    return jsonify(response_data)




def get_submenu_name(service_id, user_submenu_path):
    # Construct the full path to the JSON file using user_submenu_path
    json_file_path = os.path.join(user_submenu_path, 'menu_structure.json')
    
    # Load the JSON data from the file
    with open(json_file_path, 'r') as json_file:
        menu_data = json.load(json_file)

    # Iterate over the menu data to find the submenu with matching SubmenuId
    for menu in menu_data:
        if 'Submenu' in menu:
            for submenu in menu['Submenu']:
                if submenu['SubmenuId'] == service_id:
                    return submenu['Submenu']
@app.route("/include_service_file", methods=['POST'])
def include_service_file():
    response_data = {"success": False, "message": ""}
    try:
        file = request.files.get('file')
        service_id = request.form['SubmenuId']
        print(service_id) 
        # Assuming 'service_id' is provided in the form data
        #service_id=(service_id))
        print(service_id)
        if file and service_id:
            # Check if file is allowed
            if not allowed_file(file.filename):
                response_data["message"] = "Only PDF files are allowed"
                return jsonify(response_data)

            # Check file size
            if get_file_size(file) > MAX_FILE_SIZE:
                response_data["message"] = "File size exceeds maximum limit (10 MB)"
                return jsonify(response_data)

            # Initialize cursor after establishing connection
            connection = get_database_connection()
            cursor = connection.cursor()

            token = request.headers.get('Authorization')

            # Check if the token starts with 'Bearer ' and extract the actual token
            if token and token.startswith('Bearer '):
                actual_token = token.split('Bearer ')[1]
            else:
                # Handle the case when the token is not in the expected format
                actual_token = None
                response_data["message"] = "Invalid token format"
                cursor.close()
                connection.close()
                return jsonify(response_data)

            # Retrieve the user's ID from the Customers table based on the token
            get_user_id_query = 'SELECT user_id, first_name FROM Customers WHERE api_token = %s'
            cursor.execute(get_user_id_query, (actual_token,))
            user_id_result = cursor.fetchone()

            if user_id_result:
                user_id, first_name = user_id_result
                user_folder_name = f"{first_name}_{user_id}"

                # Retrieve the user's directory name from the Directory table
                get_directory_query = 'SELECT directory_name FROM Directory WHERE user_id = %s'
                cursor.execute(get_directory_query, (user_id,))
                directory_info = cursor.fetchone()
                if directory_info:
                    directory_name = directory_info[0]
                    user_directory_path = os.path.join(BASE_UPLOAD_FOLDER, directory_name, f"service{service_id}")
                    os.makedirs(user_directory_path, exist_ok=True)
                    user_submenu_path = os.path.join(BASE_UPLOAD_FOLDER, directory_name,"setting_pdf")
                    submenuname=get_submenu_name(service_id,user_submenu_path)
                    print(submenuname)

                    filename = secure_filename(file.filename)
                    path = os.path.join(user_directory_path, filename)
                    file.save(path)
                    print("sub=",service_id)
                    insert_filename_query = 'INSERT INTO FileLogs (user_id, file_name) VALUES (%s, %s) ON DUPLICATE KEY UPDATE user_id = VALUES(user_id), file_name = VALUES(file_name)'

                    cursor.execute(insert_filename_query, (user_id, filename))
       #             cursor.execute(insert_filename_query, (user_id, filename))
                    
                    upsert_file_query = '''
    INSERT INTO reports (file_name, type, uploaded_date, status, user_id, trained, submenu_id,extracted_prompt,deleted,uploaded,submenuname)
    VALUES (%s, 'File', CURRENT_TIMESTAMP, 'Enabled', %s, 'No', %s,'No','No','Yes',%s)
    ON DUPLICATE KEY UPDATE uploaded_date = CURRENT_TIMESTAMP, status = 'Enabled'
'''
                    cursor.execute(upsert_file_query, (filename, user_id, service_id,submenuname))

                    connection.commit()  # Commit the transaction

                    response_data["success"] = True
                    response_data["message"] = "File uploaded successfully"
                else:
                    response_data["message"] = "User directory not found"
            else:
                response_data["message"] = "Invalid token"

            cursor.close()
            connection.close()

        else:
            response_data["message"] = "No file or service ID provided"

    except Exception as e:
        send_error_email('include_service_file', str(e))
        print(f"Error: {e}")
        response_data["message"] = "An error occurred while processing the request"

    return jsonify(response_data)




def get_submenu_name_url(service_id, user_submenu_path):
    # Construct the full path to the JSON file using user_submenu_path
    json_file_path = os.path.join(user_submenu_path, 'menu_structure.json')

    # Load the JSON data from the file
    with open(json_file_path, 'r') as json_file:
        menu_data = json.load(json_file)

    # Iterate over the menu data to find the submenu with matching SubmenuId
    for menu in menu_data:
        if 'Submenu' in menu:
            for submenu in menu['Submenu']:
                if submenu['SubmenuId'] == service_id:
                    return submenu['Submenu']




@app.route("/include_url_service", methods=['POST'])
def include_url_service():
    response_data = {"success": False, "message": ""}
    try:
        include_url = request.form.get('include-url')
        service_number = request.form['SubmenuId']
        print(service_number)
       # service_number =int(float(service_number))
        if include_url:
            # Initialize cursor after establishing connection
            connection = get_database_connection()
            cursor = connection.cursor()

            token = request.headers.get('Authorization')

            # Check if the token starts with 'Bearer ' and extract the actual token
            if token and token.startswith('Bearer '):
                actual_token = token.split('Bearer ')[1]
            else:
                # Handle the case when the token is not in the expected format
                actual_token = None
                response_data["message"] = "Invalid token format"
                cursor.close()
                connection.close()
                return jsonify(response_data)

            # Retrieve the user's ID from the Customers table based on the token
            get_user_id_query = 'SELECT user_id, first_name FROM Customers WHERE api_token = %s'
            cursor.execute(get_user_id_query, (actual_token,))
            user_id_result = cursor.fetchone()

            if user_id_result:
                user_id, first_name = user_id_result
                user_folder_name = f"{first_name}_{user_id}"

                # Retrieve the user's directory name from the Directory table
                get_directory_query = 'SELECT directory_name FROM Directory WHERE user_id = %s'
                cursor.execute(get_directory_query, (user_id,))
                directory_info = cursor.fetchone()

                if directory_info:
                    directory_name = directory_info[0]
                    service_name = f"service{service_number}"
                    user_directory_path = os.path.join(BASE_UPLOAD_FOLDER, directory_name,service_name)
                    os.makedirs(user_directory_path, exist_ok=True)
                    print(user_directory_path)
                    user_submenu_path = os.path.join(BASE_UPLOAD_FOLDER, directory_name,"setting_pdf")
                    submenuname=get_submenu_name_url(service_number,user_submenu_path)
                    print(submenuname)
                    crawl_website_service1_1(include_url, user_directory_path)
                    upsert_file_query = '''
    INSERT INTO url_reports (url, type, uploaded_date, status, user_id, trained, submenu_id, extracted_prompt, deleted, uploaded, submenuname)
    VALUES (%s, 'url', CURRENT_TIMESTAMP, 'Enabled', %s, 'No', %s, 'No', 'No', 'Yes', %s)
    ON DUPLICATE KEY UPDATE 
        uploaded_date = CURRENT_TIMESTAMP, 
        status = 'Enabled',
        submenu_id = IF(submenu_id = VALUES(submenu_id), submenu_id, submenu_id),
        url = IF(user_id = VALUES(user_id), url, VALUES(url))
'''
                    cursor.execute(upsert_file_query, (include_url, user_id, service_number, submenuname))

                    #upsert_file_query = '''
                     #   INSERT INTO url_reports (url, type, uploaded_date, status, user_id, trained, submenu_id, extracted_prompt, deleted, uploaded,submenuname)
                      #  VALUES (%s, 'url', CURRENT_TIMESTAMP, 'Enabled', %s, 'No', %s, 'No', 'No', 'Yes',%s)
                       # ON DUPLICATE KEY UPDATE uploaded_date = CURRENT_TIMESTAMP, status = 'Enabled'
                   # '''
                    #cursor.execute(upsert_file_query, (include_url, user_id, service_number,submenuname))

                    # Insert URL into WebURLlog table
                    insert_url_query = 'INSERT INTO WebURLlog (user_id, url) VALUES (%s, %s) ON DUPLICATE KEY UPDATE user_id = VALUES(user_id), url = VALUES(url)'
                    cursor.execute(insert_url_query, (user_id, include_url))
                    
                    connection.commit()

                    response_data["success"] = True
                    response_data["message"] = "URL added successfully"
                else:
                    response_data["message"] = "User directory not found"
            else:
                response_data["message"] = "Invalid token"

            cursor.close()
            connection.close()

        else:
            response_data["message"] = "No URL provided"

    except Exception as e:
        send_error_email('include_url_service', str(e))
        print(f"Error: {e}")
        response_data["message"] = "An error occurred while processing the request"

    return jsonify(response_data)



@app.route('/get_file_reports', methods=['GET'])
def get_file_reports():
    try:
        # Initialize the database connection and cursor
        connection = get_database_connection()
        cursor = connection.cursor()

        # Retrieve the Authorization token from the request headers
        token = request.headers.get('Authorization')

        # Check if the token is missing or not in the correct format
        if not token or not token.startswith('Bearer '):
            return jsonify({'error': 'Unauthorized'}), 401

        # Extract the actual token from the Authorization header
        actual_token = token.split('Bearer ')[1]

        # Query the database to get the user ID associated with the token
        user_query = 'SELECT user_id FROM Customers WHERE api_token = %s'
        cursor.execute(user_query, (actual_token,))
        user_id_result = cursor.fetchone()

        # Check if the user ID was found
        if not user_id_result:
            return jsonify({'error': 'User not found'}), 404

        user_id = user_id_result[0]

        # Query the database to get the data from the reports table associated with the user ID
        file_query = 'SELECT * FROM reports WHERE user_id = %s'
        cursor.execute(file_query, (user_id,))
        file_data = cursor.fetchall()
        print(file_data)

        # Check if any data was found for the user
        if not file_data:
                return jsonify({"Reports": []})


        # Close cursor and connection
        cursor.close()
        connection.close()
    
        # Return the file data as a JSON response
        return jsonify({'Reports': [{'File ID': row[0], 'Filename': row[1], 'Type': row[2], 'Uploaded Date': row[3],
                                     'Deleted Date': row[4], 'Trained Date': row[5], 'Extract Prompt Date': row[6],
                                     'Trained': row[7], 'Extracted Prompt': row[8], 'Status': row[9],
                                     'User ID': row[10], 'Submenu ID': row[11],"Submenuname":row[14]} for row in file_data]}), 200
    except mysql.connector.Error as err:
        return jsonify({'error': f'Database error: {err}'}), 500
    except Exception as e:
        send_error_email('get_file_reports', str(e))
        # Handle any other unexpected errors
        return jsonify({'error': f'An error occurred: {e}'}), 500

















@app.route('/get_url_reports', methods=['GET'])
def get_url_reports():
    try:
        # Initialize the database connection and cursor
        connection = get_database_connection()
        cursor = connection.cursor()

        # Retrieve the Authorization token from the request headers
        token = request.headers.get('Authorization')

        # Check if the token is missing or not in the correct format
        if not token or not token.startswith('Bearer '):
            return jsonify({'error': 'Unauthorized'}), 401

        # Extract the actual token from the Authorization header
        actual_token = token.split('Bearer ')[1]

        # Query the database to get the user ID associated with the token
        user_query = 'SELECT user_id FROM Customers WHERE api_token = %s'
        cursor.execute(user_query, (actual_token,))
        user_id_result = cursor.fetchone()

        # Check if the user ID was found
        if not user_id_result:
            return jsonify({'error': 'User not found'}), 404

        user_id = user_id_result[0]

        # Query the database to get the data from the reports table associated with the user ID
        file_query = 'SELECT * FROM url_reports WHERE user_id = %s'
        cursor.execute(file_query, (user_id,))
        file_data = cursor.fetchall()

        # Check if any data was found for the user
        if not file_data:
                return jsonify({"Reports": []})


        # Close cursor and connection
        cursor.close()
        connection.close()

        # Return the file data as a JSON response
        return jsonify({'Reports': [{'Url ID': row[0], 'Url': row[1], 'Type': row[2], 'Uploaded Date': row[3],
                                     'Deleted Date': row[4], 'Trained Date': row[5], 'Extract Prompt Date': row[6],
                                     'Trained': row[7], 'Extracted Prompt': row[8], 'Status': row[9],
                                     'User ID': row[10], 'Submenu ID': row[11],"Submenuname":row[14]} for row in file_data]}), 200
    except mysql.connector.Error as err:
        return jsonify({'error': f'Database error: {err}'}), 500
    except Exception as e:
        send_error_email('get_url_reports', str(e))
        # Handle any other unexpected errors
        return jsonify({'error': f'An error occurred: {e}'}), 500















def get_data_and_chroma_paths_service(service_id):
    try:
        token = request.headers.get('Authorization')
        if not token or not token.startswith('Bearer '):
            return None, None, "Invalid token format"

        actual_token = token.split('Bearer ')[1]

        connection = get_database_connection()
        cursor = connection.cursor()

        get_user_id_query = 'SELECT user_id FROM Customers WHERE api_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()

        if not user_id_result:
            return None, None, "Invalid user ID"

        user_id = user_id_result[0]

        get_directory_query = 'SELECT directory_name FROM Directory WHERE user_id = %s'
        cursor.execute(get_directory_query, (user_id,))
        directory_info = cursor.fetchone()

        if not directory_info:
            return None, None, "User directory not found"

        directory_name = directory_info[0]
        data_path = os.path.join(BASE_UPLOAD_FOLDER, directory_name, f"service{service_id}")
        chroma_path = os.path.join(BASE_UPLOAD_FOLDER, directory_name, f"service{service_id}_chromadb")



        cursor.close()
        connection.close()

        return data_path, chroma_path, None

    except Exception as e:
        return None, None, f"An error occurred: {str(e)}"


def generate_data_store_service(service_id):
    try:
        data_path, chroma_path, error= get_data_and_chroma_paths_service(service_id)
        if error:
            return jsonify({"error": error}), 500
        documents = load_documents_service(data_path)
        chunks = split_text_service(documents)
        save_to_chroma_service(chunks, chroma_path)
        print(service_id)
        store_file_info(service_id,data_path)


        return jsonify({"message": "Data store generated successfully!"})

    except Exception as e:
        send_error_email('generate_data_store_service', str(e))
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


def load_documents_service(data_path):
    documents = []
    for filename in os.listdir(data_path):
        if filename.endswith(".pdf"):
            full_path = os.path.join(data_path, filename)
            text = extract_text_from_pdf_service(full_path)
            document = Document(filename=filename, page_content=text)
            documents.append(document)
    return documents


def extract_text_from_pdf_service(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text


def split_text_service(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    for document in chunks:
        try:
            print(document.page_content)
        except UnicodeEncodeError:
            print("UnicodeEncodeError occurred, skipping printing for this document.")

        print(document.metadata)

    return chunks


def save_to_chroma_service(chunks, chroma_path):
    os.makedirs(chroma_path,exist_ok=True)

    try:
        db = Chroma.from_documents(chunks, OpenAIEmbeddings(), persist_directory=chroma_path)
        db.persist()
        print(f"Saved {len(chunks)} chunks to {chroma_path}.")
    except Exception as e:
        print(f"Error occurred while saving to Chroma: {e}")




def store_file_info(service_id,data_path):
    try:
        token = request.headers.get('Authorization')
        if not token or not token.startswith('Bearer '):
            return jsonify({"error": "Invalid token format"}), 401

        actual_token = token.split('Bearer ')[1]

        connection = get_database_connection()
        cursor = connection.cursor()

        get_user_id_query = 'SELECT user_id FROM Customers WHERE api_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()

        if not user_id_result:
            return jsonify({"error": "Invalid user ID"}), 403

        user_id = user_id_result[0]
        print("info=",user_id)
        # Upsert file information
        upsert_file_query = '''
    INSERT INTO reports (file_name, trained,trained_date, user_id, submenu_id)
    VALUES (%s, 'No',NULL, %s, %s)
    ON DUPLICATE KEY UPDATE trained = 'Yes', trained_date = CURRENT_TIMESTAMP
'''
        for filename in os.listdir(data_path):
            if filename.endswith(".pdf"):
                print(filename)
                cursor.execute(upsert_file_query, (filename, user_id, service_id))

        connection.commit()

        cursor.close()
        connection.close()

    except Exception as e:
        return jsonify({"error": f"An error occurred while storing file info: {str(e)}"}), 500

#import signal

#def restart_server():
 #   print("Restarting server...")
  #  os.kill(os.getpid(), signal.SIGINT)

#@app.route('/restart', methods=['POST'])
#def restart():
    # Trigger the restart_server function when a POST request is made to this endpoint
  #  if request.method == 'POST':
 #       restart_server()
   #     return 'Server restarting...'
   # else:
    #    return 'Method not allowed', 405
import os
import signal
import subprocess

def restart_server():
    print("Restarting server...")
    os.execl(sys.executable, sys.executable, *sys.argv)

@app.route('/restart', methods=['POST'])
def restart():
    # Trigger the restart_server function when a POST request is made to this endpoint
    if request.method == 'POST':
        print("Received POST request to restart server")
        restart_server()
        return 'Server restarting...'
    else:
        print("Method not allowed")
        return 'Method not allowed', 405



@app.route("/generate_data_store_service", methods=["POST"])
def generate_data_store_services():
    try:
        service_id = request.form.get('SubmenuId')
        print(service_id)


    except ValueError:
        return jsonify({"error": "Invalid service ID"}), 400

    return generate_data_store_service(service_id)














def get_data_and_chroma_paths_url_service(service_id, url):
    try:
        token = request.headers.get('Authorization')
        if not token or not token.startswith('Bearer '):
            return None, None, None, "Invalid token format"

        actual_token = token.split('Bearer ')[1]

        connection = get_database_connection()
        cursor = connection.cursor()

        get_user_id_query = 'SELECT user_id FROM Customers WHERE api_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()

        if not user_id_result:
            return None, None, None, "Invalid user ID"

        user_id = user_id_result[0]

        get_directory_query = 'SELECT directory_name FROM Directory WHERE user_id = %s'
        cursor.execute(get_directory_query, (user_id,))
        directory_info = cursor.fetchone()

        if not directory_info:
            return None, None, None, "User directory not found"

        directory_name = directory_info[0]
        data_path = os.path.join(BASE_UPLOAD_FOLDER, directory_name, f"service{service_id}")
        chroma_path = os.path.join(BASE_UPLOAD_FOLDER, directory_name, f"service{service_id}_chromadb")

        cursor.close()
        connection.close()

        return data_path, chroma_path, url, None

    except Exception as e:
        return None, None, None, f"An error occurred: {str(e)}"

def generate_data_store_url_service(service_id,url):
    try:
        data_path, chroma_path,url ,error= get_data_and_chroma_paths_url_service(service_id,url)
        if error:
            return jsonify({"error": error}), 500

        documents = load_documents_url_service(data_path)
        chunks = split_text_url_service(documents)
        save_to_chroma_url_service(chunks, chroma_path)
        print(service_id)
        store_file_url_info(service_id,data_path,url)

        return jsonify({"message": "Data store generated successfully!"})

    except Exception as e:
        send_error_email('generate_data_store_url_service', str(e))
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


def load_documents_url_service(data_path):
    documents = []
    for filename in os.listdir(data_path):
        if filename.endswith(".pdf"):
            full_path = os.path.join(data_path, filename)
            text = extract_text_from_pdf_url_service(full_path)
            document = Document(filename=filename, page_content=text)
            documents.append(document)
    return documents

def extract_text_from_pdf_url_service(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text


def split_text_url_service(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    for document in chunks:
        try:
            print(document.page_content)
        except UnicodeEncodeError:
            print("UnicodeEncodeError occurred, skipping printing for this document.")

        print(document.metadata)

    return chunks


def save_to_chroma_url_service(chunks, chroma_path):
    os.makedirs(chroma_path, exist_ok=True)

    try:
        db = Chroma.from_documents(chunks, OpenAIEmbeddings(), persist_directory=chroma_path)
        db.persist()
        print(f"Saved {len(chunks)} chunks to {chroma_path}.")
    except Exception as e:
        print(f"Error occurred while saving to Chroma: {e}")
def convert_to_url_format(filename):
    # Replace underscores with slashes and remove the ".pdf" extension
    url = filename.replace("___", "://").replace(".pdf", "").replace("_", ".").replace("-", "/")
    
    # Replace triple slashes with a single slash after "https:"
    
    return url

def store_file_url_info(service_id,data_pathi,url):
    try:
        token = request.headers.get('Authorization')
        if not token or not token.startswith('Bearer '):
            return jsonify({"error": "Invalid token format"}), 401

        actual_token = token.split('Bearer ')[1]

        connection = get_database_connection()
        cursor = connection.cursor()

        get_user_id_query = 'SELECT user_id FROM Customers WHERE api_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()

        if not user_id_result:
            return jsonify({"error": "Invalid user ID"}), 403

        user_id = user_id_result[0]
        print("info=",user_id)
       

        upsert_file_query = '''
INSERT INTO url_reports (url,trained, trained_date, user_id)
VALUES (%s,'No', NULL, %s)
ON DUPLICATE KEY UPDATE trained = 'Yes', trained_date = CURRENT_TIMESTAMP;
'''


        # Upsert file information
       # upsert_file_query = '''
    #INSERT INTO url_reports (url, trained,trained_date, user_id, submenu_id)
    #VALUES (%s, 'No',NULL, %s, %s)
    #ON DUPLICATE KEY UPDATE trained = 'Yes', trained_date = CURRENT_TIMESTAMP
#'''
        cursor.execute(upsert_file_query, (url,user_id))

        connection.commit()

        cursor.close()
        connection.close()

    except Exception as e:
        return jsonify({"error": f"An error occurred while storing file info: {str(e)}"}), 500

@app.route("/generate_data_store_url_service", methods=["POST"])
def generate_data_store_url_services():
    try:
        service_id = request.form.get('SubmenuId')
        url= request.form.get('Url')
        print(service_id)

    except ValueError:
        return jsonify({"error": "Invalid service ID"}), 400

    return generate_data_store_url_service(service_id,url)












def get_data_and_chroma_paths_service_url(service_id):
    try:
        token = request.headers.get('Authorization')
        if not token or not token.startswith('Bearer '):
            return None, None, "Invalid token format"

        actual_token = token.split('Bearer ')[1]

        connection = get_database_connection()
        cursor = connection.cursor()

        get_user_id_query = 'SELECT user_id FROM Customers WHERE api_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()

        if not user_id_result:
            return None, None, "Invalid user ID"

        user_id = user_id_result[0]

        get_directory_query = 'SELECT directory_name FROM Directory WHERE user_id = %s'
        cursor.execute(get_directory_query, (user_id,))
        directory_info = cursor.fetchone()

        if not directory_info:
            return None, None, "User directory not found"

        directory_name = directory_info[0]
        data_path = os.path.join(BASE_UPLOAD_FOLDER, directory_name, f"service{service_id}")
        chroma_path = os.path.join(BASE_UPLOAD_FOLDER, directory_name, f"service{service_id}_chromadb")

        cursor.close()
        connection.close()

        return data_path, chroma_path, None

    except Exception as e:
        return None, None, f"An error occurred: {str(e)}"


def load_urls_service_url(data_path):
    with open(data_path, 'r') as file:
        urls = file.readlines()
    return urls


def extract_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = ' '.join([p.text for p in soup.find_all('p')])
    return text


def split_text_service_url(urls):
    chunks = []
    for url in urls:
        text = extract_text_from_url(url)
        if text:
            document = Document(page_content=text)  # Provide the page_content
            chunks.append(document)
        else:
            print(f"Failed to extract text from URL: {url}")
            # Optionally, you can append a placeholder document or take other actions
            # to indicate the failure without halting the process.
    return chunks


def save_to_chroma_service_url(chunks, chroma_path):
    os.makedirs(chroma_path, exist_ok=True)

    try:
        if not chunks:
            print("Error: No chunks provided.")
            return

        db = Chroma.from_documents(chunks, OpenAIEmbeddings(), persist_directory=chroma_path)
        db.persist()
        print(f"Saved {len(chunks)} chunks to {chroma_path}.")
    except Exception as e:
        print(f"Error occurred while saving to Chroma: {e}")


def store_file_info_url(service_id, data_path):
    try:
        token = request.headers.get('Authorization')
        if not token or not token.startswith('Bearer '):
            return jsonify({"error": "Invalid token format"}), 401

        actual_token = token.split('Bearer ')[1]

        connection = get_database_connection()
        cursor = connection.cursor()

        get_user_id_query = 'SELECT user_id FROM Customers WHERE api_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()

        if not user_id_result:
            return jsonify({"error": "Invalid user ID"}), 403

        user_id = user_id_result[0]

        # Upsert file information
        upsert_file_query = '''
            INSERT INTO reports (file_name, trained, status, trained_date, user_id, submenu_id)
            VALUES (%s, 'no', 'uploaded', NULL, %s, %s)
            ON DUPLICATE KEY UPDATE trained = 'yes', status = 'trained', trained_date = CURRENT_TIMESTAMP
        '''
        for url in os.listdir(data_path):
            cursor.execute(upsert_file_query, (url, user_id, service_id))

        connection.commit()

        cursor.close()
        connection.close()

    except Exception as e:
        return jsonify({"error": f"An error occurred while storing file info: {str(e)}"}), 500


@app.route("/generate_data_store_service_url", methods=["POST"])
def generate_data_store_services_url():
    try:
        # Get service ID from the form data
        service_id = request.form.get('SubmenuId')
        print(service_id)

        # Check if service ID is missing
        if service_id is None:
            return jsonify({"error": "Service ID is missing in the request"}), 400

        # Convert service ID to integer
        service_id =service_id

        # Get URLs from form data
        urls = request.form.getlist('urls')
        print(urls)

        # Check if URLs are missing
        if not urls:
            return jsonify({"error": "No URLs provided"}), 400

        # Get data and chroma paths
        data_path, chroma_path, error = get_data_and_chroma_paths_service_url(service_id)
        if error:
            return jsonify({"error": error}), 500

        # Load URLs, split text, save to chroma
        chunks = split_text_service_url(urls)
        save_to_chroma_service_url(chunks, chroma_path)

        # Store file info
        store_file_info_url(service_id, data_path)

        return jsonify({"message": "Data store generated successfully!"})

    except Exception as e:
        send_error_email('generate_data_store_service_url', str(e))
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500




@app.route("/include_url_service1", methods=['POST'])
def include_url_service1():
    response_data = {"success": False, "message": ""}
    try:
        include_url = request.form.get('include-url')
        if include_url:
            # Initialize cursor after establishing connection
            connection = get_database_connection()
            cursor = connection.cursor()

            token = request.headers.get('Authorization')

            # Check if the token starts with 'Bearer ' and extract the actual token
            if token and token.startswith('Bearer '):
                actual_token = token.split('Bearer ')[1]
            else:
                # Handle the case when the token is not in the expected format
                actual_token = None
                response_data["message"] = "Invalid token format"
                cursor.close()
                connection.close()
                return jsonify(response_data)

            # Retrieve the user's ID from the Customers table based on the token
            get_user_id_query = 'SELECT user_id, first_name FROM Customers WHERE api_token = %s'
            cursor.execute(get_user_id_query, (actual_token,))
            user_id_result = cursor.fetchone()

            if user_id_result:
                user_id, first_name = user_id_result
                user_folder_name = f"{first_name}_{user_id}"

                # Retrieve the user's directory name from the Directory table
                get_directory_query = 'SELECT directory_name FROM Directory WHERE user_id = %s'
                cursor.execute(get_directory_query, (user_id,))
                directory_info = cursor.fetchone()

                if directory_info:
                    directory_name = directory_info[0]
                    user_directory_path = os.path.join(BASE_UPLOAD_FOLDER, directory_name, "service1")
                    os.makedirs(user_directory_path, exist_ok=True)

                    crawl_website_service1(include_url, user_directory_path)

                    url = include_url
                    insert_url_query = 'INSERT INTO WebURLlog (user_id,url) VALUES (%s, %s) ON DUPLICATE KEY UPDATE user_id = VALUES(user_id), url = VALUES(url)'

                    cursor.execute(insert_url_query, (user_id, url))
                    connection.commit()

                    response_data["success"] = True
                    response_data["message"] = "URL added successfully"
                else:
                    response_data["message"] = "User directory not found"
            else:
                response_data["message"] = "Invalid token"

            cursor.close()
            connection.close()

        else:
            response_data["message"] = "No URL provided"

    except Exception as e:
        send_error_email('include_url_service1', str(e))
        print(f"Error: {e}")
        response_data["message"] = "An error occurred while processing the request"

    return jsonify(response_data)














































def get_data_and_chroma_paths_service1():
    connection = get_database_connection()
    cursor = connection.cursor()
    try:
        token = request.headers.get('Authorization')
        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
        else:
            actual_token = None
            response_data = {"message": "Invalid token format"}
            return None, None

        get_user_id_query = 'SELECT user_id FROM Customers WHERE api_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()

        if user_id_result:
            user_id = user_id_result[0]

            get_directory_query = 'SELECT directory_name FROM Directory WHERE user_id = %s'
            cursor.execute(get_directory_query, (user_id,))
            directory_info = cursor.fetchone()

            if directory_info:
                directory_name = directory_info[0]
                user_directory_path = os.path.join(BASE_UPLOAD_FOLDER, directory_name, "service1")
                data_path = user_directory_path
                user_directory=os.path.join(BASE_UPLOAD_FOLDER, directory_name,"service1_chromadb")
                chroma_path = user_directory
                return data_path, chroma_path
    except Exception as e:
        print("An error occurred:", e)
    finally:
        cursor.close()
        connection.close()
    return None, None
@app.route("/generate_data_store_service1", methods=["POST"])
def generate_data_store_service1():
    data_path, chroma_path = get_data_and_chroma_paths_service1()
    if data_path and chroma_path:
        documents = load_documents_service1(data_path)
        chunks = split_text_service1(documents)
        save_to_chroma_service1(chunks)
        return jsonify({"message": "Data store generated successfully!"})
    else:
        return jsonify({"error": "Failed to retrieve data and chroma paths."}), 500

def load_documents_service1(data_path):
    documents = []
    for filename in os.listdir(data_path):
        if filename.endswith(".pdf"):
            full_path = os.path.join(data_path, filename)
            text = extract_text_from_pdf_service1(full_path)
            document = Document(filename=filename, page_content=text)
            documents.append(document)
    return documents

def extract_text_from_pdf_service1(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def split_text_service1(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    for document in chunks:
        try:
            print(document.page_content)
        except UnicodeEncodeError:
            print("UnicodeEncodeError occurred, skipping printing for this document.")

        print(document.metadata)

    return chunks

def save_to_chroma_service1(chunks: list[Document]):
    data_path, chroma_path = get_data_and_chroma_paths_service1()
    print(chroma_path)
    os.makedirs(chroma_path, exist_ok=True)

    # Create a new DB from the documents.
    try:
        db = Chroma.from_documents(chunks, OpenAIEmbeddings(), persist_directory=chroma_path)
        db.persist()
        print(f"Saved {len(chunks)} chunks to {chroma_path}.")
    except Exception as e:
        print(f"Error occurred while saving to Chroma: {e}")







def get_data_and_chroma_paths_service2():
    connection = get_database_connection()
    cursor = connection.cursor()
    try:
        token = request.headers.get('Authorization')
        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
        else:
            actual_token = None
            response_data = {"message": "Invalid token format"}
            return None, None

        get_user_id_query = 'SELECT user_id FROM Customers WHERE api_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()

        if user_id_result:
            user_id = user_id_result[0]

            get_directory_query = 'SELECT directory_name FROM Directory WHERE user_id = %s'
            cursor.execute(get_directory_query, (user_id,))
            directory_info = cursor.fetchone()

            if directory_info:
                directory_name = directory_info[0]
                user_directory_path = os.path.join(BASE_UPLOAD_FOLDER, directory_name, "service2")
                data_path = user_directory_path
                user_directory=os.path.join(BASE_UPLOAD_FOLDER, directory_name,"service2_chromadb")
                chroma_path = user_directory
                return data_path, chroma_path
    except Exception as e:
        print("An error occurred:", e)
    finally:
        cursor.close()
        connection.close()
    return None, None
@app.route("/generate_data_store_service2", methods=["POST"])
def generate_data_store_service2():
    data_path, chroma_path = get_data_and_chroma_paths_service2()
    if data_path and chroma_path:
        documents = load_documents_service2(data_path)
        chunks = split_text_service2(documents)
        save_to_chroma_service2(chunks)
        return jsonify({"message": "Data store generated successfully!"})
    else:
        return jsonify({"error": "Failed to retrieve data and chroma paths."}), 500

def load_documents_service2(data_path):
    documents = []
    for filename in os.listdir(data_path):
        if filename.endswith(".pdf"):
            full_path = os.path.join(data_path, filename)
            text = extract_text_from_pdf_service2(full_path)
            document = Document(filename=filename, page_content=text)
            documents.append(document)
    return documents

def extract_text_from_pdf_service2(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def split_text_service2(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    for document in chunks:
        try:
            print(document.page_content)
        except UnicodeEncodeError:
            print("UnicodeEncodeError occurred, skipping printing for this document.")

        print(document.metadata)

    return chunks

def save_to_chroma_service2(chunks: list[Document]):
    data_path, chroma_path = get_data_and_chroma_paths_service2()
    print(chroma_path)
    os.makedirs(chroma_path, exist_ok=True)

    # Create a new DB from the documents.
    try:
        db = Chroma.from_documents(chunks, OpenAIEmbeddings(), persist_directory=chroma_path)
        db.persist()
        print(f"Saved {len(chunks)} chunks to {chroma_path}.")
    except Exception as e:
        print(f"Error occurred while saving to Chroma: {e}")






def get_data_and_chroma_paths_service3():
    connection = get_database_connection()
    cursor = connection.cursor()
    try:
        token = request.headers.get('Authorization')
        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
        else:
            actual_token = None
            response_data = {"message": "Invalid token format"}
            return None, None

        get_user_id_query = 'SELECT user_id FROM Customers WHERE api_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()

        if user_id_result:
            user_id = user_id_result[0]

            get_directory_query = 'SELECT directory_name FROM Directory WHERE user_id = %s'
            cursor.execute(get_directory_query, (user_id,))
            directory_info = cursor.fetchone()

            if directory_info:
                directory_name = directory_info[0]
                user_directory_path = os.path.join(BASE_UPLOAD_FOLDER, directory_name, "service3")
                data_path = user_directory_path
                user_directory=os.path.join(BASE_UPLOAD_FOLDER, directory_name,"service3_chromadb")
                chroma_path = user_directory
                return data_path, chroma_path
    except Exception as e:
        print("An error occurred:", e)
    finally:
        cursor.close()
        connection.close()
    return None, None
@app.route("/generate_data_store_service3", methods=["POST"])
def generate_data_store_service3():
    data_path, chroma_path = get_data_and_chroma_paths_service3()
    if data_path and chroma_path:
        documents = load_documents_service3(data_path)
        chunks = split_text_service3(documents)
        save_to_chroma_service3(chunks)
        return jsonify({"message": "Data store generated successfully!"})
    else:
        return jsonify({"error": "Failed to retrieve data and chroma paths."}), 500

def load_documents_service3(data_path):
    documents = []
    for filename in os.listdir(data_path):
        if filename.endswith(".pdf"):
            full_path = os.path.join(data_path, filename)
            text = extract_text_from_pdf_service3(full_path)
            document = Document(filename=filename, page_content=text)
            documents.append(document)
    return documents

def extract_text_from_pdf_service3(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def split_text_service3(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    for document in chunks:
        try:
            print(document.page_content)
        except UnicodeEncodeError:
            print("UnicodeEncodeError occurred, skipping printing for this document.")

        print(document.metadata)

    return chunks

def save_to_chroma_service3(chunks: list[Document]):
    data_path, chroma_path = get_data_and_chroma_paths_service3()
    print(chroma_path)
    os.makedirs(chroma_path, exist_ok=True)

    # Create a new DB from the documents.
    try:
        db = Chroma.from_documents(chunks, OpenAIEmbeddings(), persist_directory=chroma_path)
        db.persist()
        print(f"Saved {len(chunks)} chunks to {chroma_path}.")
    except Exception as e:
        print(f"Error occurred while saving to Chroma: {e}")











def get_data_and_chroma_paths_service4():
    connection = get_database_connection()
    cursor = connection.cursor()
    try:
        token = request.headers.get('Authorization')
        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
        else:
            actual_token = None
            response_data = {"message": "Invalid token format"}
            return None, None

        get_user_id_query = 'SELECT user_id FROM Customers WHERE api_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()

        if user_id_result:
            user_id = user_id_result[0]

            get_directory_query = 'SELECT directory_name FROM Directory WHERE user_id = %s'
            cursor.execute(get_directory_query, (user_id,))
            directory_info = cursor.fetchone()

            if directory_info:
                directory_name = directory_info[0]
                user_directory_path = os.path.join(BASE_UPLOAD_FOLDER, directory_name, "service4")
                data_path = user_directory_path
                user_directory=os.path.join(BASE_UPLOAD_FOLDER, directory_name,"service4_chromadb")
                chroma_path = user_directory
                return data_path, chroma_path
    except Exception as e:
        print("An error occurred:", e)
    finally:
        cursor.close()
        connection.close()
    return None, None
@app.route("/generate_data_store_service4", methods=["POST"])
def generate_data_store_service4():
    data_path, chroma_path = get_data_and_chroma_paths_service4()
    if data_path and chroma_path:
        documents = load_documents_service4(data_path)
        chunks = split_text_service4(documents)
        save_to_chroma_service4(chunks)
        return jsonify({"message": "Data store generated successfully!"})
    else:
        return jsonify({"error": "Failed to retrieve data and chroma paths."}), 500

def load_documents_service4(data_path):
    documents = []
    for filename in os.listdir(data_path):
        if filename.endswith(".pdf"):
            full_path = os.path.join(data_path, filename)
            text = extract_text_from_pdf_service4(full_path)
            document = Document(filename=filename, page_content=text)
            documents.append(document)
    return documents

def extract_text_from_pdf_service4(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def split_text_service4(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    for document in chunks:
        try:
            print(document.page_content)
        except UnicodeEncodeError:
            print("UnicodeEncodeError occurred, skipping printing for this document.")

        print(document.metadata)

    return chunks

def save_to_chroma_service4(chunks: list[Document]):
    data_path, chroma_path = get_data_and_chroma_paths_service4()
    print(chroma_path)
    os.makedirs(chroma_path, exist_ok=True)

    # Create a new DB from the documents.
    try:
        db = Chroma.from_documents(chunks, OpenAIEmbeddings(), persist_directory=chroma_path)
        db.persist()
        print(f"Saved {len(chunks)} chunks to {chroma_path}.")
    except Exception as e:
        print(f"Error occurred while saving to Chroma: {e}")






@app.route('/header_text', methods=['POST'])
def header_text():
    connection = None
    cursor = None

    try:
        # Get the authorization token from the request headers
        token = request.headers.get('Authorization')
        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]

        # Establish database connection
        connection = get_database_connection()
        cursor = connection.cursor()

        # Retrieve user ID using the provided token
        get_user_id_query = 'SELECT user_id FROM Customers WHERE api_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()

        if not user_id_result:
            return jsonify({'error': 'User not found or invalid token'}), 404

        user_id = user_id_result[0]

        # Extract data from request body
        data = request.json
        header_text1 = data.get('headerText1')
        header_text2 = data.get('headerText2')
        print(data)

        # Upsert into introductions table
        upsert_query = '''
            INSERT INTO header_text
            (user_id, header_text1, header_text2)
            VALUES
            (%s, %s, %s)
            ON DUPLICATE KEY UPDATE
            header_text1 = VALUES(header_text1),
            header_text2 = VALUES(header_text2)
        '''
        upsert_data = (user_id, header_text1, header_text2)
        print(upsert_data)
        cursor.execute(upsert_query, upsert_data)
        connection.commit()

        return jsonify({'message': 'Introduction upserted successfully'}), 200

    except mysql.connector.Error as err:
        return jsonify({'error': f"Database error: {err}"}), 500

    except Exception as e:
        send_error_email('header_text', str(e))
        return jsonify({'error': str(e)}), 500

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()
















import re
import os
import pdfkit
import requests
from bs4 import BeautifulSoup

def crawl_website_service1_1(url, save_directory):
    try:
        # Set a custom user agent
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

        # Send a GET request to the URL with custom headers
        response = requests.get(url, headers=headers)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the HTML content of the page
            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract the text content from the webpage
            text_content = soup.get_text()

            # Process the text content
            processed_text_content = process_text_content_service1_1(text_content)

            # Save the PDF for the current page
            save_to_pdf_service1_1(processed_text_content, url, save_directory)
        else:
            print("Failed to retrieve the webpage. Status code:", response.status_code)
    except Exception as e:
        send_error_email('crawl_website_service1_1', str(e))
        print("An error occurred:", str(e))

def process_text_content_service1_1(text_content):
    # Split the text into lines
    lines = text_content.split('\n')

    # Process each line
    processed_lines = []
    current_sentence = ''
    for line in lines:
        line = line.strip()
        if line:  # Check if line is not empty
            # Check if the line ends with a punctuation indicating the end of a sentence
            if re.search(r'[.!?]$', line):
                # If it does, add it to the current sentence
                current_sentence += ' ' + line
                processed_lines.append(current_sentence.strip())  # Add completed sentence
                current_sentence = ''  # Reset current sentence
            else:
                # If it doesn't end a sentence, add it to the current sentence
                current_sentence += ' ' + line

    # Join the processed lines into paragraphs
    processed_text = '\n'.join(processed_lines)

    return processed_text

def save_to_pdf_service1_1(text_content, url, save_directory):
    try:
        # Specify the path to wkhtmltopdf
        path_to_wkhtmltopdf = '/usr/bin/wkhtmltopdf'  # Specify the path to wkhtmltopdf

        # Generate filename from URL
       # filename = url.replace('/', '_').replace(':', '_').replace('.', '_') + ".pdf"
        filename = url.replace('://', '___').replace('/', '-').replace('.', '_') + ".pdf"
        # Construct the full path to save the PDF
        filepath = os.path.join(save_directory, filename)

        # Convert the text content to PDF using pdfkit
        pdfkit.from_string(text_content, filepath, configuration=pdfkit.configuration(wkhtmltopdf=path_to_wkhtmltopdf))
        print("PDF saved successfully as", filepath)
    except Exception as e:
        print("An error occurred while saving the PDF:", str(e))







def crawl_website_service2(url, save_directory):
    try:
        # Set a custom user agent
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

        # Send a GET request to the URL with custom headers
        response = requests.get(url, headers=headers)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the HTML content of the page
            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract the text content from the webpage
            text_content = soup.get_text()

            # Process the text content
            processed_text_content = process_text_content_service2(text_content)

            # Save the PDF for the current page
            save_to_pdf_service2(processed_text_content, url, save_directory)
        else:
            print("Failed to retrieve the webpage. Status code:", response.status_code)
    except Exception as e:
        print("An error occurred:", str(e))

def process_text_content_service2(text_content):
    # Split the text into paragraphs
    paragraphs = text_content.split('\n\n')

    # Process each paragraph
    processed_paragraphs = []
    for paragraph in paragraphs:
        # Split the paragraph into sentences
        sentences = re.split(r'(?<=[.!?]) +', paragraph)

        # Join the sentences with proper formatting
        processed_paragraph = ' '.join(sentences)

        # Append the processed paragraph to the list
        processed_paragraphs.append(processed_paragraph)

    # Join the processed paragraphs with proper formatting
    processed_text = '\n\n'.join(processed_paragraphs)

    return processed_text

def save_to_pdf_service2(text_content, url, save_directory):
    try:
        # Specify the path to wkhtmltopdf
        path_to_wkhtmltopdf = '/usr/bin/wkhtmltopdf'  # Specify the path to wkhtmltopdf

        # Generate filename from URL
        filename = url.replace('/', '_').replace(':', '_').replace('.', '_') + ".pdf"

        # Construct the full path to save the PDF
        filepath = os.path.join(save_directory, filename)

        # Convert the text content to PDF using pdfkit
        pdfkit.from_string(text_content, filepath, configuration=pdfkit.configuration(wkhtmltopdf=path_to_wkhtmltopdf))
        print("PDF saved successfully as", filepath)
    except Exception as e:
        print("An error occurred while saving the PDF:", str(e))







def crawl_website_service3(url, save_directory):
    try:
        # Set a custom user agent
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

        # Send a GET request to the URL with custom headers
        response = requests.get(url, headers=headers)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the HTML content of the page
            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract the text content from the webpage
            text_content = soup.get_text()

            # Process the text content
            processed_text_content = process_text_content_service3(text_content)

            # Save the PDF for the current page
            save_to_pdf_service3(processed_text_content, url, save_directory)
        else:
            print("Failed to retrieve the webpage. Status code:", response.status_code)
    except Exception as e:
        print("An error occurred:", str(e))

def process_text_content_service3(text_content):
    # Split the text into paragraphs
    paragraphs = text_content.split('\n\n')

    # Process each paragraph
    processed_paragraphs = []
    for paragraph in paragraphs:
        # Split the paragraph into sentences
        sentences = re.split(r'(?<=[.!?]) +', paragraph)

        # Join the sentences with proper formatting
        processed_paragraph = ' '.join(sentences)

        # Append the processed paragraph to the list
        processed_paragraphs.append(processed_paragraph)

    # Join the processed paragraphs with proper formatting
    processed_text = '\n\n'.join(processed_paragraphs)

    return processed_text

def save_to_pdf_service3(text_content, url, save_directory):
    try:
        # Specify the path to wkhtmltopdf
        path_to_wkhtmltopdf = '/usr/bin/wkhtmltopdf'  # Specify the path to wkhtmltopdf

        # Generate filename from URL
        filename = url.replace('/', '_').replace(':', '_').replace('.', '_') + ".pdf"

        # Construct the full path to save the PDF
        filepath = os.path.join(save_directory, filename)

        # Convert the text content to PDF using pdfkit
        pdfkit.from_string(text_content, filepath, configuration=pdfkit.configuration(wkhtmltopdf=path_to_wkhtmltopdf))
        print("PDF saved successfully as", filepath)
    except Exception as e:
        print("An error occurred while saving the PDF:", str(e))







def crawl_website_service4(url, save_directory):
    try:
        # Set a custom user agent
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

        # Send a GET request to the URL with custom headers
        response = requests.get(url, headers=headers)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the HTML content of the page
            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract the text content from the webpage
            text_content = soup.get_text()

            # Process the text content
            processed_text_content = process_text_content_service4(text_content)

            # Save the PDF for the current page
            save_to_pdf_service4(processed_text_content, url, save_directory)
        else:
            print("Failed to retrieve the webpage. Status code:", response.status_code)
    except Exception as e:
        print("An error occurred:", str(e))

def process_text_content_service4(text_content):
    # Split the text into paragraphs
    paragraphs = text_content.split('\n\n')

    # Process each paragraph
    processed_paragraphs = []
    for paragraph in paragraphs:
        # Split the paragraph into sentences
        sentences = re.split(r'(?<=[.!?]) +', paragraph)

        # Join the sentences with proper formatting
        processed_paragraph = ' '.join(sentences)

        # Append the processed paragraph to the list
        processed_paragraphs.append(processed_paragraph)

    # Join the processed paragraphs with proper formatting
    processed_text = '\n\n'.join(processed_paragraphs)

    return processed_text

def save_to_pdf_service4(text_content, url, save_directory):
    try:
        # Specify the path to wkhtmltopdf
        path_to_wkhtmltopdf = '/usr/bin/wkhtmltopdf'  # Specify the path to wkhtmltopdf

        # Generate filename from URL
        filename = url.replace('/', '_').replace(':', '_').replace('.', '_') + ".pdf"

        # Construct the full path to save the PDF
        filepath = os.path.join(save_directory, filename)

        # Convert the text content to PDF using pdfkit
        pdfkit.from_string(text_content, filepath, configuration=pdfkit.configuration(wkhtmltopdf=path_to_wkhtmltopdf))
        print("PDF saved successfully as", filepath)
    except Exception as e:
        print("An error occurred while saving the PDF:", str(e))











@app.route('/header_text', methods=['GET'])
def get_header_text():
    connection = None
    cursor = None

    try:
        # Get the authorization token from the request headers
        token = request.headers.get('Authorization')
        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]

        # Establish database connection
        connection = get_database_connection()
        cursor = connection.cursor()

        # Retrieve user ID using the provided token
        get_user_id_query = 'SELECT user_id FROM Customers WHERE user_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()

        if not user_id_result:
            return jsonify({'error': 'User not found or invalid token'}), 404

        user_id = user_id_result[0]

        # Retrieve introduction and descriptions for the user
        get_introduction_query = 'SELECT header_text1, header_text2 FROM header_text WHERE user_id = %s'
        cursor.execute(get_introduction_query, (user_id,))
        introduction_result = cursor.fetchone()
        print(introduction_result)

        if not introduction_result:
            return jsonify({'error': 'Introduction not found for the user'}), 404

        header_text1, header_text2 = introduction_result

        return jsonify({
            'header_text1': header_text1,
            'header_text2': header_text2
        }), 200

    except mysql.connector.Error as err:
        return jsonify({'error': f"Database error: {err}"}), 500

    except Exception as e:
        send_error_email('header_text_get', str(e))
        return jsonify({'error': str(e)}), 500

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()











MAX_IMAGES = 3  # Maximum number of images to keep in the directory

@app.route('/upload_image_chatbot', methods=['POST'])
def upload_image_chatbot():
    try:
        token = request.headers.get('Authorization')
        print("Token:", token)

        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
            print(actual_token)

            # Retrieve user_id and directory name from the database using token
            connection = get_database_connection()
            cursor = connection.cursor()
            cursor.execute('SELECT user_id FROM Customers WHERE api_token = %s', (actual_token,))
            user_id_result = cursor.fetchone()

            if user_id_result:
                user_id = user_id_result[0]
                print("User ID:", user_id)

                cursor.execute('SELECT directory_name FROM Directory WHERE user_id = %s', (user_id,))
                directory_info = cursor.fetchone()
                print(directory_info)

                if directory_info:
                    directory_name = directory_info[0]
                    print("Directory:", directory_name)

                    # Check if the post request has the file part
                    if 'file' not in request.files:
                        return jsonify({'error': 'No file part in the request'}), 400

                    # Retrieve files from the request
                    files = request.files.getlist('file')

                    # Check if there are any files
                    if not files:
                        return jsonify({'error': 'No file uploaded'}), 400
                    # Check if the file is one of the allowed types/extensions
                    for file in files:
                        filename = secure_filename(file.filename)
                        ext = os.path.splitext(filename)[1].lower()
                        if ext not in ['.png', '.jpg', '.jpeg', '.gif']:
                           return jsonify({'error': 'Invalid file type. Allowed file types are: png, jpg, jpeg, gif'}), 400

                    # Iterate over files

                    save_folder = os.path.join(BASE_UPLOAD_FOLDER, directory_name, "chatbot_images")
                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)

                    # Remove old images if they exist
                    for i in range(1, MAX_IMAGES + 1):
                        old_image_path = os.path.join(save_folder, f"image{i}{ext}")
                        if os.path.exists(old_image_path):
                            os.remove(old_image_path)

                    # Save the received image files to the specified directory
                    # Use a single name for each image file
                    for index, file in enumerate(files, start=1):
                        filename = f"image{index}{ext}"
                        file.save(os.path.join(save_folder, filename))

                    return jsonify({'message': 'Images uploaded successfully'})
                else:
                    return jsonify({'error': 'Directory not found for the user'}), 404
            else:
                return jsonify({'error': 'User not found'}), 404
        else:
            return jsonify({'error': 'Invalid token'}), 401
    except Exception as e:
        send_error_email('upload_image_chatbot', str(e))
        return jsonify({'error': str(e)}), 500
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

@app.route('/get_images_chatbot', methods=['GET'])
def get_images_chatbot():
    try:
        token = request.headers.get('Authorization')
        print("Token:", token)

        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
            print(actual_token)

            # Retrieve user_id and directory name from the database using token
            connection = get_database_connection()
            cursor = connection.cursor()
            cursor.execute('SELECT user_id FROM Customers WHERE api_token = %s', (actual_token,))
            user_id_result = cursor.fetchone()

            if user_id_result:
                user_id = user_id_result[0]
                print("User ID:", user_id)

                cursor.execute('SELECT directory_name FROM Directory WHERE user_id = %s', (user_id,))
                directory_info = cursor.fetchone()
                print(directory_info)

                if directory_info:
                    directory_name = directory_info[0]
                    print("Directory:", directory_name)

                    # Define the directory path to retrieve images
                    image_folder = os.path.join(BASE_UPLOAD_FOLDER, directory_name, "chatbot_images")

                    # Check if the directory exists
                    if not os.path.exists(image_folder):
                        return jsonify({'error': 'Directory not found'}), 404

                    # Retrieve all image filenames from the directory
                    images = os.listdir(image_folder)

                    # Return the list of image filenames
                    return jsonify({'images': images})

                else:
                    return jsonify({'error': 'Directory not found for the user'}), 404
            else:
                return jsonify({'error': 'User not found'}), 404
        else:
            return jsonify({'error': 'Invalid token'}), 401
    except Exception as e:
        send_error_email('get_images_chatbot', str(e))
        return jsonify({'error': str(e)}), 500
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


@app.route("/include_source_service1", methods=['POST'])
def include_source_service1():
    response_data = {"success": False, "message": ""}
    try:
        file = request.files.get('file')
        include_url = request.form.get('include-url')

        if file:
            # Check if file is allowed
            if not allowed_file(file.filename):
                response_data["message"] = "Only PDF files are allowed"
                print(response_data)
                return jsonify(response_data)

            # Check file size
            if get_file_size(file) > MAX_FILE_SIZE:
                response_data["message"] = "File size exceeds maximum limit (10 MB)"
                return jsonify(response_data)

        # Initialize cursor after establishing connection
        connection = get_database_connection()
        cursor = connection.cursor()

        token = request.headers.get('Authorization')

        # Check if the token starts with 'Bearer ' and extract the actual token
        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
        else:
            # Handle the case when the token is not in the expected format
            actual_token = None
            response_data["message"] = "Invalid token format"
            cursor.close()
            connection.close()
            return jsonify(response_data)

        # Retrieve the user's ID from the Customers table based on the token
        get_user_id_query = 'SELECT user_id, first_name FROM Customers WHERE api_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()

        if user_id_result:
            user_id, first_name = user_id_result
            user_folder_name = f"{first_name}_{user_id}"

            # Retrieve the user's directory name from the Directory table
            get_directory_query = 'SELECT directory_name FROM Directory WHERE user_id = %s'
            cursor.execute(get_directory_query, (user_id,))
            directory_info = cursor.fetchone()

            if directory_info:
                directory_name = directory_info[0]
                user_directory_path = os.path.join(BASE_UPLOAD_FOLDER, directory_name, "service1")
                os.makedirs(user_directory_path, exist_ok=True)

                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    path = os.path.join(user_directory_path, filename)
                    file.save(path)
                    insert_filename_query = 'INSERT INTO FileLogs (user_id, file_name) VALUES (%s, %s) ON DUPLICATE KEY UPDATE user_id = VALUES(user_id), file_name = VALUES(file_name)'

                    cursor.execute(insert_filename_query, (user_id, filename))
                    connection.commit()  # Commit the transaction

                    context["sources_to_add"].append(filename)

                    response_data["success"] = True
                    response_data["message"] = "File uploaded successfully"

                elif include_url:
                    save_directory = user_directory_path
                    print(save_directory)

                    crawl_website_service1(include_url, save_directory)

                    url = include_url
                    insert_url_query = 'INSERT INTO WebURLlog (user_id,url) VALUES (%s, %s) ON DUPLICATE KEY UPDATE user_id = VALUES(user_id), url = VALUES(url)'

                    cursor.execute(insert_url_query, (user_id, url))
                    connection.commit()

                    context["sources_to_add"].append(include_url)
                    response_data["success"] = True
                    response_data["message"] = "URL added successfully"
                else:
                    response_data["message"] = "Invalid file or URL"
            else:
                response_data["message"] = "User directory not found"
        else:
            response_data["message"] = "Invalid token"

        cursor.close()
        connection.close()

    except Exception as e:
        print(f"Error: {e}")
        response_data["message"] = "An error occurred while processing the request"

    return jsonify(response_data)



@app.route("/include_source_service2", methods=['POST'])
def include_source_service2():
    response_data = {"success": False, "message": ""}
    try:
        file = request.files.get('file')
        include_url = request.form.get('include-url')

        if file:
            # Check if file is allowed
            if not allowed_file(file.filename):
                response_data["message"] = "Only PDF files are allowed"
                print(response_data)
                return jsonify(response_data)

            # Check file size
            if get_file_size(file) > MAX_FILE_SIZE:
                response_data["message"] = "File size exceeds maximum limit (10 MB)"
                return jsonify(response_data)

        # Initialize cursor after establishing connection
        connection = get_database_connection()
        cursor = connection.cursor()

        token = request.headers.get('Authorization')

        # Check if the token starts with 'Bearer ' and extract the actual token
        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
        else:
            # Handle the case when the token is not in the expected format
            actual_token = None
            response_data["message"] = "Invalid token format"
            cursor.close()
            connection.close()
            return jsonify(response_data)

        # Retrieve the user's ID from the Customers table based on the token
        get_user_id_query = 'SELECT user_id, first_name FROM Customers WHERE api_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()

        if user_id_result:
            user_id, first_name = user_id_result
            user_folder_name = f"{first_name}_{user_id}"

            # Retrieve the user's directory name from the Directory table
            get_directory_query = 'SELECT directory_name FROM Directory WHERE user_id = %s'
            cursor.execute(get_directory_query, (user_id,))
            directory_info = cursor.fetchone()

            if directory_info:
                directory_name = directory_info[0]
                user_directory_path = os.path.join(BASE_UPLOAD_FOLDER, directory_name, "service2")
                os.makedirs(user_directory_path, exist_ok=True)

                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    path = os.path.join(user_directory_path, filename)
                    file.save(path)
                    insert_filename_query = 'INSERT INTO FileLogs (user_id, file_name) VALUES (%s, %s) ON DUPLICATE KEY UPDATE user_id = VALUES(user_id), file_name = VALUES(file_name)'

                    cursor.execute(insert_filename_query, (user_id, filename))
                    connection.commit()  # Commit the transaction

                    context["sources_to_add"].append(filename)

                    response_data["success"] = True
                    response_data["message"] = "File uploaded successfully"

                elif include_url:
                    save_directory = user_directory_path
                    print(save_directory)

                    crawl_website_service2(include_url, save_directory)

                    url = include_url
                    insert_url_query = 'INSERT INTO WebURLlog (user_id,url) VALUES (%s, %s) ON DUPLICATE KEY UPDATE user_id = VALUES(user_id), url = VALUES(url)'

                    cursor.execute(insert_url_query, (user_id, url))
                    connection.commit()

                    context["sources_to_add"].append(include_url)
                    response_data["success"] = True
                    response_data["message"] = "URL added successfully"
                else:
                    response_data["message"] = "Invalid file or URL"
            else:
                response_data["message"] = "User directory not found"
        else:
            response_data["message"] = "Invalid token"

        cursor.close()
        connection.close()

    except Exception as e:
        print(f"Error: {e}")
        response_data["message"] = "An error occurred while processing the request"

    return jsonify(response_data)



@app.route("/include_source_service3", methods=['POST'])
def include_source_service3():
    response_data = {"success": False, "message": ""}
    try:
        file = request.files.get('file')
        include_url = request.form.get('include-url')

        if file:
            # Check if file is allowed
            if not allowed_file(file.filename):
                response_data["message"] = "Only PDF files are allowed"
                print(response_data)
                return jsonify(response_data)

            # Check file size
            if get_file_size(file) > MAX_FILE_SIZE:
                response_data["message"] = "File size exceeds maximum limit (10 MB)"
                return jsonify(response_data)

        # Initialize cursor after establishing connection
        connection = get_database_connection()
        cursor = connection.cursor()

        token = request.headers.get('Authorization')

        # Check if the token starts with 'Bearer ' and extract the actual token
        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
        else:
            # Handle the case when the token is not in the expected format
            actual_token = None
            response_data["message"] = "Invalid token format"
            cursor.close()
            connection.close()
            return jsonify(response_data)

        # Retrieve the user's ID from the Customers table based on the token
        get_user_id_query = 'SELECT user_id, first_name FROM Customers WHERE api_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()

        if user_id_result:
            user_id, first_name = user_id_result
            user_folder_name = f"{first_name}_{user_id}"

            # Retrieve the user's directory name from the Directory table
            get_directory_query = 'SELECT directory_name FROM Directory WHERE user_id = %s'
            cursor.execute(get_directory_query, (user_id,))
            directory_info = cursor.fetchone()

            if directory_info:
                directory_name = directory_info[0]
                user_directory_path = os.path.join(BASE_UPLOAD_FOLDER, directory_name, "service3")
                os.makedirs(user_directory_path, exist_ok=True)

                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    path = os.path.join(user_directory_path, filename)
                    file.save(path)
                    insert_filename_query = 'INSERT INTO FileLogs (user_id, file_name) VALUES (%s, %s) ON DUPLICATE KEY UPDATE user_id = VALUES(user_id), file_name = VALUES(file_name)'

                    cursor.execute(insert_filename_query, (user_id, filename))
                    connection.commit()  # Commit the transaction

                    context["sources_to_add"].append(filename)

                    response_data["success"] = True
                    response_data["message"] = "File uploaded successfully"

                elif include_url:
                    save_directory = user_directory_path
                    print(save_directory)

                    crawl_website_service3(include_url, save_directory)

                    url = include_url
                    insert_url_query = 'INSERT INTO WebURLlog (user_id,url) VALUES (%s, %s) ON DUPLICATE KEY UPDATE user_id = VALUES(user_id), url = VALUES(url)'

                    cursor.execute(insert_url_query, (user_id, url))
                    connection.commit()

                    context["sources_to_add"].append(include_url)
                    response_data["success"] = True
                    response_data["message"] = "URL added successfully"
                else:
                    response_data["message"] = "Invalid file or URL"
            else:
                response_data["message"] = "User directory not found"
        else:
            response_data["message"] = "Invalid token"

        cursor.close()
        connection.close()

    except Exception as e:
        print(f"Error: {e}")
        response_data["message"] = "An error occurred while processing the request"

    return jsonify(response_data)



@app.route("/include_source_service4", methods=['POST'])
def include_source_service4():
    response_data = {"success": False, "message": ""}
    try:
        file = request.files.get('file')
        include_url = request.form.get('include-url')

        if file:
            # Check if file is allowed
            if not allowed_file(file.filename):
                response_data["message"] = "Only PDF files are allowed"
                print(response_data)
                return jsonify(response_data)

            # Check file size
            if get_file_size(file) > MAX_FILE_SIZE:
                response_data["message"] = "File size exceeds maximum limit (10 MB)"
                return jsonify(response_data)

        # Initialize cursor after establishing connection
        connection = get_database_connection()
        cursor = connection.cursor()

        token = request.headers.get('Authorization')

        # Check if the token starts with 'Bearer ' and extract the actual token
        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
        else:
            # Handle the case when the token is not in the expected format
            actual_token = None
            response_data["message"] = "Invalid token format"
            cursor.close()
            connection.close()
            return jsonify(response_data)

        # Retrieve the user's ID from the Customers table based on the token
        get_user_id_query = 'SELECT user_id, first_name FROM Customers WHERE api_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()

        if user_id_result:
            user_id, first_name = user_id_result
            user_folder_name = f"{first_name}_{user_id}"

            # Retrieve the user's directory name from the Directory table
            get_directory_query = 'SELECT directory_name FROM Directory WHERE user_id = %s'
            cursor.execute(get_directory_query, (user_id,))
            directory_info = cursor.fetchone()

            if directory_info:
                directory_name = directory_info[0]
                user_directory_path = os.path.join(BASE_UPLOAD_FOLDER, directory_name, "service4")
                os.makedirs(user_directory_path, exist_ok=True)

                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    path = os.path.join(user_directory_path, filename)
                    file.save(path)
                    insert_filename_query = 'INSERT INTO FileLogs (user_id, file_name) VALUES (%s, %s) ON DUPLICATE KEY UPDATE user_id = VALUES(user_id), file_name = VALUES(file_name)'

                    cursor.execute(insert_filename_query, (user_id, filename))
                    connection.commit()  # Commit the transaction

                    context["sources_to_add"].append(filename)

                    response_data["success"] = True
                    response_data["message"] = "File uploaded successfully"

                elif include_url:
                    save_directory = user_directory_path
                    print(save_directory)

                    crawl_website_service4(include_url, save_directory)

                    url = include_url
                    insert_url_query = 'INSERT INTO WebURLlog (user_id,url) VALUES (%s, %s) ON DUPLICATE KEY UPDATE user_id = VALUES(user_id), url = VALUES(url)'

                    cursor.execute(insert_url_query, (user_id, url))
                    connection.commit()

                    context["sources_to_add"].append(include_url)
                    response_data["success"] = True
                    response_data["message"] = "URL added successfully"
                else:
                    response_data["message"] = "Invalid file or URL"
            else:
                response_data["message"] = "User directory not found"
        else:
            response_data["message"] = "Invalid token"

        cursor.close()
        connection.close()

    except Exception as e:
        print(f"Error: {e}")
        response_data["message"] = "An error occurred while processing the request"

    return jsonify(response_data)






def get_data_and_chroma_path_web1():
    connection = get_database_connection()  # Assuming you have a function to establish a database connection
    cursor = connection.cursor()
    try:

       token = request.headers.get('Authorization')
       if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
       else:
            actual_token = None
            response_data = {"message": "Invalid token format"}
            return None, None

       get_user_id_query = 'SELECT user_id FROM Customers WHERE user_token = %s'
       cursor.execute(get_user_id_query, (actual_token,))
       user_id_result = cursor.fetchone()

       if user_id_result:
            user_id = user_id_result[0]

       get_directory_query = 'SELECT directory_name FROM Directory WHERE user_id = %s'
       cursor.execute(get_directory_query, (user_id,))
       directory_info = cursor.fetchone()

       if directory_info:
                directory_name = directory_info[0]
                user_directory_path = os.path.join(BASE_UPLOAD_FOLDER, directory_name, "service1")
                data_path = user_directory_path
                user_directory = os.path.join(BASE_UPLOAD_FOLDER, directory_name,'service1_chromadb')
                chroma_path = user_directory
                print(chroma_path)
                return data_path, chroma_path
    except Exception as e:
        print("An error occurred:", e)
    finally:
        cursor.close()
        connection.close()
    return None, None
@app.route("/query_web1", methods=["POST"])
def query_chroma_web1():
    PROMPT_TEMPLATE = """


    # You will be acting as an AI PDF Expert named GentAI.
    # Your goal is to provide accurate answers and insights based on the given context.
    # You will be replying to users who may be confused if you don't respond appropriately.
    # You are provided with a PDF document for context.

    Answer the question based only on the following context:
    GentAI, an AI PDF Expert, provides assistance based on the given context.
    To get started, please follow these steps:
    1. Briefly introduce yourself as the AI PDF Expert GentAI.
    2. Describe the content.
    ---
    {context}

    ---

    Answer the question based on the above context: {question}
    """

#    data = request.get_json()
 #   print(data)
  #  query_text = data['question']

   # print(query_text)

    # Get data and chroma paths dynamically
    data_path, chroma_path = get_data_and_chroma_path_web1()
    print(chroma_path)
    data = request.json
    query_text = data['question']
    print(query_text)
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)

    print(db)
    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    print(results)
    if len(results) == 0 or results[0][1] < 0.7: #or #results[0][1] < 0.5:
        # If no matching results found, answer the question directly.
        model = ChatOpenAI()
        response_text = model.predict(query_text)
        response_text = "\n\nI am GentAI, the AI PDF Expert." + response_text

        sources = []  # No sources available since it's not based on context

        return jsonify({"response_text": response_text, "sources": sources})


    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    print(context_text)
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt_kwargs = {
        "context": context_text,
        "question": query_text,
        "answer": ""  # Placeholder for the answer
    }
    prompt = prompt_template.format(**prompt_kwargs)

    model = ChatOpenAI()
    response_text = model.predict(prompt)
    print(response_text)
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = {"response_text": response_text, "sources": sources}
    print(formatted_response)
    #return jsonify(formatted_response)
   # print("data"+user_id)
    # Insert question and answer into ChatLogs table based on user_id
    connection = get_database_connection()  # Assuming you have a function to establish a database connection
    cursor = connection.cursor()
    try:

       token = request.headers.get('Authorization')
       if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
            print("Inside_web"+actual_token)
       else:
            actual_token = None
            response_data = {"message": "Invalid token format"}
            return None, None
       get_user_id_query = 'SELECT user_id FROM Customers WHERE user_token = %s'
       cursor.execute(get_user_id_query, (actual_token,))
       user_id_result = cursor.fetchone()
       user_id=user_id_result[0]
       print(user_id)
       response_string = json.dumps(formatted_response)

       insert_chat_log_query = "INSERT INTO ChatLogs (user_id, question, answer,DateTimeColumn) VALUES (%s, %s, %s,CURRENT_TIMESTAMP)"
       chat_log_data = (user_id, query_text,response_string )
       print("inside+web",chat_log_data)
       cursor.execute(insert_chat_log_query, chat_log_data)
       connection.commit()

    except Exception as e:
        print("An error occurred:", e)
    finally:
        cursor.close()
        connection.close()
    return jsonify(formatted_response)


def get_data_and_chroma_path_web2():
    connection = get_database_connection()  # Assuming you have a function to establish a database connection
    cursor = connection.cursor()
    try:

       token = request.headers.get('Authorization')
       if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
       else:
            actual_token = None
            response_data = {"message": "Invalid token format"}
            return None, None

       get_user_id_query = 'SELECT user_id FROM Customers WHERE user_token = %s'
       cursor.execute(get_user_id_query, (actual_token,))
       user_id_result = cursor.fetchone()

       if user_id_result:
            user_id = user_id_result[0]

       get_directory_query = 'SELECT directory_name FROM Directory WHERE user_id = %s'
       cursor.execute(get_directory_query, (user_id,))
       directory_info = cursor.fetchone()

       if directory_info:
                directory_name = directory_info[0]
                user_directory_path = os.path.join(BASE_UPLOAD_FOLDER, directory_name, "service2")
                data_path = user_directory_path
                user_directory = os.path.join(BASE_UPLOAD_FOLDER, directory_name,'service2_chromadb')
                chroma_path = user_directory
                print(chroma_path)
                return data_path, chroma_path
    except Exception as e:
        print("An error occurred:", e)
    finally:
        cursor.close()
        connection.close()
    return None, None
@app.route("/query_web2", methods=["POST"])
def query_chroma_web2():
    PROMPT_TEMPLATE = """


    # You will be acting as an AI PDF Expert named GentAI.
    # Your goal is to provide accurate answers and insights based on the given context.
    # You will be replying to users who may be confused if you don't respond appropriately.
    # You are provided with a PDF document for context.

    Answer the question based only on the following context:
    GentAI, an AI PDF Expert, provides assistance based on the given context.
    To get started, please follow these steps:
    1. Briefly introduce yourself as the AI PDF Expert GentAI.
    2. Describe the content.
    ---
    {context}

    ---

    Answer the question based on the above context: {question}
    """

#    data = request.get_json()
 #   print(data)
  #  query_text = data['question']

   # print(query_text)

    # Get data and chroma paths dynamically
    data_path, chroma_path = get_data_and_chroma_path_web2()
    print(chroma_path)
    data = request.json
    query_text = data['question']
    print(query_text)
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)

    print(db)
    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    print(results)
    if len(results) == 0 or results[0][1] < 0.7: #or #results[0][1] < 0.5:
        # If no matching results found, answer the question directly.
        model = ChatOpenAI()
        response_text = model.predict(query_text)
        response_text = "\n\nI am GentAI, the AI PDF Expert." + response_text

        sources = []  # No sources available since it's not based on context

        return jsonify({"response_text": response_text, "sources": sources})


    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    print(context_text)
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt_kwargs = {
        "context": context_text,
        "question": query_text,
        "answer": ""  # Placeholder for the answer
    }
    prompt = prompt_template.format(**prompt_kwargs)

    model = ChatOpenAI()
    response_text = model.predict(prompt)
    print(response_text)
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = {"response_text": response_text, "sources": sources}
    print(formatted_response)
    #return jsonify(formatted_response)
   # print("data"+user_id)
    # Insert question and answer into ChatLogs table based on user_id
    connection = get_database_connection()  # Assuming you have a function to establish a database connection
    cursor = connection.cursor()
    try:

       token = request.headers.get('Authorization')
       if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
            print("Inside_web"+actual_token)
       else:
            actual_token = None
            response_data = {"message": "Invalid token format"}
            return None, None
       get_user_id_query = 'SELECT user_id FROM Customers WHERE user_token = %s'
       cursor.execute(get_user_id_query, (actual_token,))
       user_id_result = cursor.fetchone()
       user_id=user_id_result[0]
       print(user_id)
       response_string = json.dumps(formatted_response)

       insert_chat_log_query = "INSERT INTO ChatLogs (user_id, question, answer,DateTimeColumn) VALUES (%s, %s, %s,CURRENT_TIMESTAMP)"
       chat_log_data = (user_id, query_text,response_string )
       print("inside+web",chat_log_data)
       cursor.execute(insert_chat_log_query, chat_log_data)
       connection.commit()

    except Exception as e:
        print("An error occurred:", e)
    finally:
        cursor.close()
        connection.close()
    return jsonify(formatted_response)

def get_data_and_chroma_path_web3():
    connection = get_database_connection()  # Assuming you have a function to establish a database connection
    cursor = connection.cursor()
    try:

       token = request.headers.get('Authorization')
       if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
       else:
            actual_token = None
            response_data = {"message": "Invalid token format"}
            return None, None

       get_user_id_query = 'SELECT user_id FROM Customers WHERE user_token = %s'
       cursor.execute(get_user_id_query, (actual_token,))
       user_id_result = cursor.fetchone()

       if user_id_result:
            user_id = user_id_result[0]

       get_directory_query = 'SELECT directory_name FROM Directory WHERE user_id = %s'
       cursor.execute(get_directory_query, (user_id,))
       directory_info = cursor.fetchone()

       if directory_info:
                directory_name = directory_info[0]
                user_directory_path = os.path.join(BASE_UPLOAD_FOLDER, directory_name, "service3")
                data_path = user_directory_path
                user_directory = os.path.join(BASE_UPLOAD_FOLDER, directory_name,'service3_chromadb')
                chroma_path = user_directory
                print(chroma_path)
                return data_path, chroma_path
    except Exception as e:
        print("An error occurred:", e)
    finally:
        cursor.close()
        connection.close()
    return None, None
@app.route("/query_web3", methods=["POST"])
def query_chroma_web3():
    PROMPT_TEMPLATE = """


    # You will be acting as an AI PDF Expert named GentAI.
    # Your goal is to provide accurate answers and insights based on the given context.
    # You will be replying to users who may be confused if you don't respond appropriately.
    # You are provided with a PDF document for context.

    Answer the question based only on the following context:
    GentAI, an AI PDF Expert, provides assistance based on the given context.
    To get started, please follow these steps:
    1. Briefly introduce yourself as the AI PDF Expert GentAI.
    2. Describe the content.
    ---
    {context}

    ---

    Answer the question based on the above context: {question}
    """

#    data = request.get_json()
 #   print(data)
  #  query_text = data['question']

   # print(query_text)

    # Get data and chroma paths dynamically
    data_path, chroma_path = get_data_and_chroma_path_web3()
    print(chroma_path)
    data = request.json
    query_text = data['question']
    print(query_text)
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)

    print(db)
    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    print(results)
    if len(results) == 0 or results[0][1] < 0.7: #or #results[0][1] < 0.5:
        # If no matching results found, answer the question directly.
        model = ChatOpenAI()
        response_text = model.predict(query_text)
        response_text = "\n\nI am GentAI, the AI PDF Expert." + response_text

        sources = []  # No sources available since it's not based on context

        return jsonify({"response_text": response_text, "sources": sources})


    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    print(context_text)
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt_kwargs = {
        "context": context_text,
        "question": query_text,
        "answer": ""  # Placeholder for the answer
    }
    prompt = prompt_template.format(**prompt_kwargs)

    model = ChatOpenAI()
    response_text = model.predict(prompt)
    print(response_text)
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = {"response_text": response_text, "sources": sources}
    print(formatted_response)
    #return jsonify(formatted_response)
   # print("data"+user_id)
    # Insert question and answer into ChatLogs table based on user_id
    connection = get_database_connection()  # Assuming you have a function to establish a database connection
    cursor = connection.cursor()
    try:

       token = request.headers.get('Authorization')
       if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
            print("Inside_web"+actual_token)
       else:
            actual_token = None
            response_data = {"message": "Invalid token format"}
            return None, None
       get_user_id_query = 'SELECT user_id FROM Customers WHERE user_token = %s'
       cursor.execute(get_user_id_query, (actual_token,))
       user_id_result = cursor.fetchone()
       user_id=user_id_result[0]
       print(user_id)
       response_string = json.dumps(formatted_response)

       insert_chat_log_query = "INSERT INTO ChatLogs (user_id, question, answer,DateTimeColumn) VALUES (%s, %s, %s,CURRENT_TIMESTAMP)"
       chat_log_data = (user_id, query_text,response_string )
       print("inside+web",chat_log_data)
       cursor.execute(insert_chat_log_query, chat_log_data)
       connection.commit()

    except Exception as e:
        print("An error occurred:", e)
    finally:
        cursor.close()
        connection.close()
    return jsonify(formatted_response)




def get_data_and_chroma_path_web4():
    connection = get_database_connection()  # Assuming you have a function to establish a database connection
    cursor = connection.cursor()
    try:

       token = request.headers.get('Authorization')
       if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
       else:
            actual_token = None
            response_data = {"message": "Invalid token format"}
            return None, None

       get_user_id_query = 'SELECT user_id FROM Customers WHERE user_token = %s'
       cursor.execute(get_user_id_query, (actual_token,))
       user_id_result = cursor.fetchone()

       if user_id_result:
            user_id = user_id_result[0]

       get_directory_query = 'SELECT directory_name FROM Directory WHERE user_id = %s'
       cursor.execute(get_directory_query, (user_id,))
       directory_info = cursor.fetchone()

       if directory_info:
                directory_name = directory_info[0]
                user_directory_path = os.path.join(BASE_UPLOAD_FOLDER, directory_name, "service4")
                data_path = user_directory_path
                user_directory = os.path.join(BASE_UPLOAD_FOLDER, directory_name,'service4_chromadb')
                chroma_path = user_directory
                print(chroma_path)
                return data_path, chroma_path
    except Exception as e:
        print("An error occurred:", e)
    finally:
        cursor.close()
        connection.close()
    return None, None
@app.route("/query_web4", methods=["POST"])
def query_chroma_web4():
    PROMPT_TEMPLATE = """


    # You will be acting as an AI PDF Expert named GentAI.
    # Your goal is to provide accurate answers and insights based on the given context.
    # You will be replying to users who may be confused if you don't respond appropriately.
    # You are provided with a PDF document for context.

    Answer the question based only on the following context:
    GentAI, an AI PDF Expert, provides assistance based on the given context.
    To get started, please follow these steps:
    1. Briefly introduce yourself as the AI PDF Expert GentAI.
    2. Describe the content.
    ---
    {context}

    ---

    Answer the question based on the above context: {question}
    """

#    data = request.get_json()
 #   print(data)
  #  query_text = data['question']

   # print(query_text)

    # Get data and chroma paths dynamically
    data_path, chroma_path = get_data_and_chroma_path_web4()
    print(chroma_path)
    data = request.json
    query_text = data['question']
    print(query_text)
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)

    print(db)
    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    print(results)
    if len(results) == 0 or results[0][1] < 0.7: #or #results[0][1] < 0.5:
        # If no matching results found, answer the question directly.
        model = ChatOpenAI()
        response_text = model.predict(query_text)
        response_text = "\n\nI am GentAI, the AI PDF Expert." + response_text

        sources = []  # No sources available since it's not based on context

        return jsonify({"response_text": response_text, "sources": sources})


    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    print(context_text)
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt_kwargs = {
        "context": context_text,
        "question": query_text,
        "answer": ""  # Placeholder for the answer
    }
    prompt = prompt_template.format(**prompt_kwargs)

    model = ChatOpenAI()
    response_text = model.predict(prompt)
    print(response_text)
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = {"response_text": response_text, "sources": sources}
    print(formatted_response)
    #return jsonify(formatted_response)
   # print("data"+user_id)
    # Insert question and answer into ChatLogs table based on user_id
    connection = get_database_connection()  # Assuming you have a function to establish a database connection
    cursor = connection.cursor()
    try:

       token = request.headers.get('Authorization')
       if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
            print("Inside_web"+actual_token)
       else:
            actual_token = None
            response_data = {"message": "Invalid token format"}
            return None, None
       get_user_id_query = 'SELECT user_id FROM Customers WHERE user_token = %s'
       cursor.execute(get_user_id_query, (actual_token,))
       user_id_result = cursor.fetchone()
       user_id=user_id_result[0]
       print(user_id)
       response_string = json.dumps(formatted_response)

       insert_chat_log_query = "INSERT INTO ChatLogs (user_id, question, answer,DateTimeColumn) VALUES (%s, %s, %s,CURRENT_TIMESTAMP)"
       chat_log_data = (user_id, query_text,response_string )
       print("inside+web",chat_log_data)
       cursor.execute(insert_chat_log_query, chat_log_data)
       connection.commit()

    except Exception as e:
        print("An error occurred:", e)
    finally:
        cursor.close()
        connection.close()
    return jsonify(formatted_response)












def get_data_and_chroma_path_web_service_admin(service_number):
    connection = get_database_connection()  # Assuming you have a function to establish a database connection
    cursor = connection.cursor()
    try:

        token = request.headers.get('Authorization')
        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
            print(actual_token)
        else:
            actual_token = None
            response_data = {"message": "Invalid token format"}
            return None, None

        get_user_id_query = 'SELECT user_id FROM Customers WHERE api_token = %s OR user_token = %s'
        cursor.execute(get_user_id_query, (actual_token,actual_token))
        user_id_result = cursor.fetchone()

        if user_id_result:
            user_id = user_id_result[0]

        get_directory_query = 'SELECT directory_name FROM Directory WHERE user_id = %s'
        cursor.execute(get_directory_query, (user_id,))
        directory_info = cursor.fetchone()

        if directory_info:
            directory_name = directory_info[0]
            user_directory_path = os.path.join(BASE_UPLOAD_FOLDER, directory_name, f"service{service_number}")
            data_path = user_directory_path
            user_directory = os.path.join(BASE_UPLOAD_FOLDER, directory_name, f'service{service_number}_chromadb')
            chroma_path = user_directory
            prompt_directory = os.path.join(BASE_UPLOAD_FOLDER, directory_name, f'prompt{service_number}')
            prompt_path = prompt_directory
            print(chroma_path)
            return data_path, chroma_path,prompt_path,user_id
    except Exception as e:
        print("An error occurred:", e)
    finally:
        cursor.close()
        connection.close()
    return None, None,None,None
@app.route("/get_prompt_admin", methods=["POST"])
def query_chroma_web_service_admin():
    data = request.get_json()
    SubmenuId = data['SubmenuId']
    types = data['Type']
    URL = data.get('URL', None)
    Filename = data.get('Filename', None)
    #URL=ata['URL']
    #Filename=data['Filename']
    print("YRLLLLLLLLLL=",URL)
    print("YRLLLLLLLLLL=",Filename)
    promptMessage = None

    lowercaseType = types.lower()

    PROMPT_TEMPLATE = """
    # You will be acting as an AI PDF Expert named GentAI.
    # Your goal is to provide accurate answers and insights based on the given context.
    # You will be replying to users who may be confused if you don't respond appropriately.
    # You are provided with a PDF document for context.
    Answer the question based only on the following context:
    GentAI, an AI PDF Expert, provides assistance based on the given context.
    To get started, please follow these steps:
    #1. Briefly introduce yourself as the AI PDF Expert GentAI.
    #2. Describe the content.
    #1. Provide 3 example questions using bullet points.
    3. Generate 3 questions related to the content of the PDF document. Ensure that each question is detailed and specific to the context provided.

    ---
    {context}
    ---
    Answer the question based on the above context: {question}
    """

    # Get data and chroma paths dynamically
    service_number = SubmenuId
    #time.sleep(10)
    data_path, chroma_path,prompt_path,user_id = get_data_and_chroma_path_web_service_admin(service_number)
    query_text = data['Prompt']
    print("query_text"+query_text)


    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt_kwargs = {
        "context": context_text,
        "question": query_text,
        "answer": ""  # Placeholder for the answer
    }
    prompt = prompt_template.format(**prompt_kwargs)

    model = ChatOpenAI()
    response_text = model.predict(prompt)

    service_object = {"service": service_number}
    typ = {"type": types}

    if lowercaseType == 'submenu':
        promptMessage = 'Prompt'
    else:
        promptMessage = 'Prompt'

    sources = [doc.metadata.get("source", None) for doc, score in results]
    formatted_response = {"response_text": response_text, "sources": sources, **service_object, **typ}

    questions = re.findall(r'\d+\.\s*(.*\?)', response_text)
    question_list = [{
        ""
        "UniqueID": f"{SubmenuId}.{i+1}_{user_id}",
        "SubmenuId": f"{SubmenuId}",
        "Type": f"{promptMessage}",
        "PromtQuery": query_text,
        "PromtId": f"{SubmenuId}",
        "PromptValue": question.strip(),
        "Score": score
    } for i, (question, score) in enumerate(zip(questions[:3], [score for _, score in results]))]
    print(question_list)
    print("prompt_path",prompt_path)
    if Filename is not None:
       print("url_File")
       store_prompt_info_file(data_path, SubmenuId, Filename)
    elif URL is not None:
       print("url_function")
       store_prompt_info_url(data_path, SubmenuId, URL)
    else:
    
       peint("Nothing")
    os.makedirs(prompt_path, exist_ok=True)

    json_filename = "question_list.json"

    json_file_path = os.path.join(prompt_path, json_filename)

    with open(json_file_path, "w") as json_file:
        json.dump(question_list, json_file)
    return jsonify(question_list)








@app.route("/read_prompt", methods=["POST"])
def read_prompt():
    try:
        # Get service_number from request body
        data = request.get_json()
        service_number = data.get("SubmenuId")

        if service_number is None:
            return jsonify({"error": "Service number is required"}), 400

        # Get database connection
        connection = get_database_connection()
        cursor = connection.cursor()

        # Get token from request headers
        token = request.headers.get('Authorization')
        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
        else:
            return jsonify({"error": "Invalid token format"}), 401

        # Retrieve user_id from the database based on the provided token
        get_user_id_query = 'SELECT user_id FROM Customers WHERE api_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()
        if user_id_result:
            user_id = user_id_result[0]
        else:
            return jsonify({"error": "User not found for the provided token"}), 404

        # Assume BASE_UPLOAD_FOLDER is defined somewhere in your application
        # Retrieve directory_name from the database based on user_id
        get_directory_query = 'SELECT directory_name FROM Directory WHERE user_id = %s'
        cursor.execute(get_directory_query, (user_id,))
        directory_info = cursor.fetchone()
        if directory_info:
            directory_name = directory_info[0]
        else:
            return jsonify({"error": "Directory not found for the user"}), 404

        # Construct the prompt directory path
        prompt_directory = os.path.join(BASE_UPLOAD_FOLDER, directory_name, f'prompt{service_number}')

        # Construct the path to the JSON file based on the service number and user_id
        json_filename = "question_list.json"
        json_file_path = os.path.join(prompt_directory, json_filename)

        # Read the JSON file
        with open(json_file_path, "r") as json_file:
            question_list = json.load(json_file)

        # Modify the structure of question_list
        modified_question_list = []
        for i, question_data in enumerate(question_list):
            SubmenuId = question_data["SubmenuId"]
            promptMessage = question_data["Type"]
            query_text = question_data["PromtQuery"]
            question = question_data["PromptValue"].strip()
            score = question_data["Score"]

            unique_id =question_data["UniqueID"]

            modified_question_list.append({
                "UniqueID": unique_id,
                "SubmenuId": SubmenuId,
                "Type": promptMessage,
                "PromtQuery": query_text,
                "PromtId": SubmenuId,
                "PromptValue": question,
                "Score": score
            })

        # Return the modified question_list
        return jsonify(modified_question_list)
    except FileNotFoundError:
        return jsonify({"message": "Prompt data not found for the specified service number"}), 404
    except Exception as e:
        #send_error_email('read_prompt', str(e))
        return jsonify({"message": "An error occurred while processing the request"}), 500
    finally:
        cursor.close()
        connection.close()








def get_data_and_chroma_path_web_service(service_number):
    connection = get_database_connection()  # Assuming you have a function to establish a database connection
    cursor = connection.cursor()
    try:

        token = request.headers.get('Authorization')
        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
            print(actual_token)
        else:
            actual_token = None
            response_data = {"message": "Invalid token format"}
            return None, None

        get_user_id_query = 'SELECT user_id FROM Customers WHERE api_token = %s OR user_token = %s'
        cursor.execute(get_user_id_query, (actual_token,actual_token))
        user_id_result = cursor.fetchone()

        if user_id_result:
            user_id = user_id_result[0]

        get_directory_query = 'SELECT directory_name FROM Directory WHERE user_id = %s'
        cursor.execute(get_directory_query, (user_id,))
        directory_info = cursor.fetchone()

        if directory_info:
            directory_name = directory_info[0]
            user_directory_path = os.path.join(BASE_UPLOAD_FOLDER, directory_name, f"service{service_number}")
            data_path = user_directory_path
            user_directory = os.path.join(BASE_UPLOAD_FOLDER, directory_name, f'service{service_number}_chromadb')
            chroma_path = user_directory
            print(chroma_path)
            return data_path, chroma_path
    except Exception as e:
        print("An error occurred:", e)
    finally:
        cursor.close()
        connection.close()
    return None, None




import re
@app.route("/get_prompt", methods=["POST"])
def query_chroma_web_service():
    data = request.get_json()
    SubmenuId = data['SubmenuId']
    types = data['Type']
    promptMessage = None

    lowercaseType = types.lower()

    PROMPT_TEMPLATE = """
    # You will be acting as an AI PDF Expert named GentAI.
    # Your goal is to provide accurate answers and insights based on the given context.
    # You will be replying to users who may be confused if you don't respond appropriately.
    # You are provided with a PDF document for context.
    Answer the question based only on the following context:
    GentAI, an AI PDF Expert, provides assistance based on the given context.
    To get started, please follow these steps:
    #1. Briefly introduce yourself as the AI PDF Expert GentAI.
    #2. Describe the content.
    #1. Provide 3 example questions using bullet points.
    3. Generate 3 questions related to the content of the PDF document. Ensure that each question is detailed and specific to the context provided.

    ---
    {context}
    ---
    Answer the question based on the above context: {question}
    """

    # Get data and chroma paths dynamically
    service_number = SubmenuId
    data_path, chroma_path = get_data_and_chroma_path_web_service(service_number)
    query_text = data['Prompt']


    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt_kwargs = {
        "context": context_text,
        "question": query_text,
    "answer": ""  # Placeholder for the answer
    }
    prompt = prompt_template.format(**prompt_kwargs)

    model = ChatOpenAI()
    response_text = model.predict(prompt)

    service_object = {"service": service_number}
    typ = {"type": types}

    if lowercaseType == 'submenu':
        promptMessage = 'Prompt'
    else:
        promptMessage = 'Prompt'

    sources = [doc.metadata.get("source", None) for doc, score in results]
    formatted_response = {"response_text": response_text, "sources": sources, **service_object, **typ}

    questions = re.findall(r'\d+\.\s*(.*\?)', response_text)
    question_list = [{
        "SubmenuId": f"{SubmenuId}",
        "Type": f"{promptMessage}",
        "PromtQuery": query_text,
        "PromtId": f"{SubmenuId}",
        "PromptValue": question.strip(),
        "Score": score
    } for i, (question, score) in enumerate(zip(questions[:3], [score for _, score in results]))]
    #store_prompt_info(data_path, SubmenuId)
    connection = get_database_connection()
    cursor = connection.cursor()
    try:
        token = request.headers.get('Authorization')
        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
        else:
            actual_token = None
            response_data = {"message": "Invalid token format"}
            return None, None

        get_user_id_query = 'SELECT user_id FROM Customers WHERE user_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()
        user_id = user_id_result[0]

        response_string = json.dumps(formatted_response)
        insert_chat_log_query = "INSERT INTO ChatLogs (user_id, question, answer,DateTimeColumn) VALUES (%s, %s, %s,CURRENT_TIMESTAMP)"
        chat_log_data = (user_id, query_text, response_string)
        cursor.execute(insert_chat_log_query, chat_log_data)

        connection.commit()
      #  store_prompt_info(data_path, SubmenuId)

    except Exception as e:
        print("An error occurred:", e)
    finally:
        cursor.close()
        connection.close()

    return jsonify(question_list)










def convert_to_url_format(filename):
    # Replace underscores with slashes and remove the ".pdf" extension
    url = filename.replace("___", "://").replace(".pdf", "").replace("_", ".").replace("-", "/")

    # Replace triple slashes with a single slash after "https:"

    return url
def store_prompt_info_file(data_path, service_id,Filename):
    print("Insidefunc")
    connection = get_database_connection()
    cursor = connection.cursor()
    try:
        token = request.headers.get('Authorization')
        if not token or not token.startswith('Bearer '):
            return jsonify({"error": "Invalid token format"}), 401

        actual_token = token.split('Bearer ')[1]

        connection = get_database_connection()
        cursor = connection.cursor()

        get_user_id_query = 'SELECT user_id FROM Customers WHERE api_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()

        if not user_id_result:
            return jsonify({"error": "Invalid user ID"}), 403

        user_id = user_id_result[0]
        
        update_file_query = '''
    UPDATE reports
    SET extracted_prompt = 'Yes', extract_prompt_date = CURRENT_TIMESTAMP
    WHERE file_name = %s AND user_id = %s AND submenu_id=%s
'''

        cursor.execute(update_file_query, (Filename, user_id, service_id))
        connection.commit()

        cursor.close()
        connection.close()
    except Exception as e:
        return jsonify({"error": f"An error occurred while storing file info: {str(e)}"}), 500





def store_prompt_info_url(data_path, service_id, URL):
    try:
        token = request.headers.get('Authorization')
        if not token or not token.startswith('Bearer '):
            return jsonify({"error": "Invalid token format"}), 401

        actual_token = token.split('Bearer ')[1]

        connection = get_database_connection()
        cursor = connection.cursor()

        get_user_id_query = 'SELECT user_id FROM Customers WHERE api_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()

        if not user_id_result:
            return jsonify({"error": "Invalid user ID"}), 403

        user_id = user_id_result[0]

       # upsert_file_query = '''
        #    INSERT INTO files (filename, user_id, service_id)
         #   VALUES (%s, %s, %s)
          #  ON DUPLICATE KEY UPDATE filename = VALUES(filename)
        #'''

        update_url_query = '''
            UPDATE url_reports
            SET extracted_prompt = 'Yes', extract_prompt_date = CURRENT_TIMESTAMP
            WHERE url = %s AND user_id = %s AND submenu_id = %s
        '''

        for filename in os.listdir(data_path):
            if filename.endswith(".pdf") and not filename.startswith("https"):
                #cursor.execute(upsert_file_query, (Filename, user_id, service_id))
                print(filename)

        for filename in os.listdir(data_path):
            if filename.endswith(".pdf"):
                url = convert_to_url_format(filename)
                print("Processing URL:", url)
                try:
                    cursor.execute(update_url_query, (URL, user_id, service_id))
                    if cursor.rowcount == 0:
                        print("No rows updated for URL:", url)
                    else:
                        print("Successfully updated URL:", url)
                except mysql.connector.Error as err:
                    print("Error updating URL:", url, "Error:", err)

        connection.commit()

        cursor.close()
        connection.close()

    except Exception as e:
        return jsonify({"error": f"An error occurred while storing file info: {str(e)}"}), 500




@app.route('/get_menu_structure', methods=['GET'])
def get_menu_structure():
    try:
        token = request.headers.get('Authorization')
        print("Token:", token)

        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
            print(actual_token)

            # Retrieve user_id and directory name from the database using token
            connection = get_database_connection()
            cursor = connection.cursor()
            cursor.execute('SELECT user_id FROM Customers WHERE api_token = %s', (actual_token,))
            user_id_result = cursor.fetchone()

            if user_id_result:
                user_id = user_id_result[0]
                print("User ID:", user_id)

                cursor.execute('SELECT directory_name FROM Directory WHERE user_id = %s', (user_id,))
                directory_info = cursor.fetchone()
                print(directory_info)

                if directory_info:
                    directory_name = directory_info[0]
                    print("Directory:", directory_name)

                    # Path to the menu structure file
                    menu_file_path = os.path.join(BASE_UPLOAD_FOLDER, directory_name, "setting_pdf", "menu_structure.json")

                    # Check if the file exists
                    if os.path.exists(menu_file_path):
                        # Load and return the menu structure JSON
                        with open(menu_file_path, 'r') as file:
                            menu_structure = json.load(file)
                        return jsonify(menu_structure)
                    else:
                        return jsonify({'error': 'Menu structure file not found'}), 404
                else:
                    return jsonify({'error': 'Directory not found for the user'}), 404
            else:
                return jsonify({'error': 'User not found'}), 404
        else:
            return jsonify({'error': 'Invalid token'}), 401
    except Exception as e:
        send_error_email('get_menu_structure', str(e))
        return jsonify({'error': str(e)}), 500
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()
@app.route('/get_smtp_details', methods=['GET'])
def get_smtp_details():
    connection = None
    cursor = None
    try:
        connection = get_database_connection()
        cursor = connection.cursor()

        # Retrieve token from headers
        token = request.headers.get('Authorization')
        if not token or not token.startswith('Bearer '):
            return jsonify({'error': 'Invalid token'}), 401

        actual_token = token.split('Bearer ')[1]

        # Retrieve user ID based on token
        get_user_id_query = 'SELECT user_id FROM Customers WHERE api_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()

        if not user_id_result:
            return jsonify({'error': 'User not found'}), 404

        user_id = user_id_result[0]

        # Retrieve SMTP details for the user
        get_smtp_query = 'SELECT port, smtp_server, email FROM smtp_details WHERE user_id = %s'
        cursor.execute(get_smtp_query, (user_id,))
        smtp_details = cursor.fetchone()

        if not smtp_details:
            return jsonify({'error': 'SMTP details not found for the user'}), 404

        smtp_data = {
            'port': smtp_details[0],
            'smtp_server': smtp_details[1],
            'email': smtp_details[2]
        }

        return jsonify(smtp_data), 200

    except Exception as e:
        send_error_email('get_smtp_details', str(e))
        return jsonify({'error': str(e)}), 500

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


@app.route('/upsert_token', methods=['POST'])
def upsert_token():
    connection = None
    cursor = None
    try:
        # Log incoming request
        print('Received request:', request.json)

        # Establish the database connection
        connection = get_database_connection()
        cursor = connection.cursor()

        # Retrieve token from headers
        token = request.headers.get('Authorization')
        print('Authorization header:', token)  # Debugging line
        if not token or not token.startswith('Bearer '):
            return jsonify({'error': 'Invalid token'}), 401

        actual_token = token.split('Bearer ')[1]
        print('Extracted actual token:', actual_token)  # Debugging line

        # Retrieve user ID based on token
        get_user_id_query = 'SELECT user_id FROM Customers WHERE api_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()

        if not user_id_result:
            return jsonify({'error': 'User not found'}), 404

        user_id = user_id_result[0]
        print('User ID:', user_id)  # Debugging line

        # Retrieve new token from request
        data = request.json
        new_token = data.get('token')
        print('New token from request:', new_token)  # Debugging line

        if not new_token:
            return jsonify({'error': 'Token is required'}), 400

        # Upsert token for the user
        upsert_token_query = """
            INSERT INTO tokens (user_id, token)
            VALUES (%s, %s)
            ON DUPLICATE KEY UPDATE token = VALUES(token)
        """
        cursor.execute(upsert_token_query, (user_id, new_token))

        connection.commit()

        return jsonify({'message': 'Token upserted successfully'}), 200

    except Exception as e:
        if connection:
            connection.rollback()
#        send_error_email('upsert_token', str(e))
        return jsonify({'error': str(e)}), 500

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

@app.route('/get_token', methods=['GET'])
def get_token():
    connection = None
    cursor = None
    try:
        print('Received GET request for token')
        connection = get_database_connection()
        cursor = connection.cursor()
        token = request.headers.get('Authorization')
        print('Authorization header:', token)
        if not token or not token.startswith('Bearer '):
            return jsonify({'error': 'Invalid token'}), 401
        actual_token = token.split('Bearer ')[1]
        print('Extracted actual token:', actual_token)
        get_user_id_query = 'SELECT user_id FROM Customers WHERE api_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()
        if not user_id_result:
            return jsonify({'error': 'User not found'}), 404
        user_id = user_id_result[0]
        print('User ID:', user_id)
        get_token_query = 'SELECT token FROM tokens WHERE user_id = %s'
        cursor.execute(get_token_query, (user_id,))
        token_result = cursor.fetchone()
        if not token_result:
            return jsonify({'error': 'Token not found'}), 404
        user_token = token_result[0]
        return jsonify({'token': user_token}), 200
    except Exception as e:
        send_error_email('get_token', str(e))
        return jsonify({'error': str(e)}), 500
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

@app.route("/promtedite", methods=["POST"])
def promptedit():
    try:
        connection = get_database_connection()
        cursor = connection.cursor()

        data = request.get_json()
        SubmenuId = data['SubmenuId']
        Type = data['Type']
        PromptValue = data['PromptValue']
        PromtId = data['PromtId']
        PromtQuery = data['PromtQuery']
        Score = data['Score']
        Unique_Id = data['UniqueId']  # Adding unique_id from request data
        print(Unique_Id)
        print(Score)
        token = request.headers.get('Authorization')
        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
        else:
            return jsonify({"error": "Invalid token format"})

        # Retrieve user_id from the database based on the provided token
        get_user_id_query = 'SELECT user_id FROM Customers WHERE api_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()
        if user_id_result:
            user_id = user_id_result[0]
        else:
            return jsonify({"error": "User not found for the provided token"})

        # Construct the upsert query
        print(user_id)
        UniqueId=f"{Unique_Id}_{user_id}"
        upsert_query = """
            INSERT INTO QuestionData (PromptValue, PromtId, PromtQuery, Score, SubmenuId, Type, user_id, unique_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
            PromptValue = VALUES(PromptValue),
            PromtQuery = VALUES(PromtQuery),
            Score = VALUES(Score),
            Type = VALUES(Type),
            ModificationTime = CURRENT_TIMESTAMP
        """

        # Execute the upsert query
        cursor.execute(upsert_query, (PromptValue, PromtId, PromtQuery, Score, SubmenuId, Type, user_id, UniqueId))
        connection.commit()

        return jsonify({"message": "Data inserted or updated successfully"})
    except Exception as e:
        send_error_email('promtedite', str(e))
        return jsonify({"error": str(e)})
    finally:
        cursor.close()
        connection.close()


import time
@app.route('/get_file', methods=['GET'])
def get_file():
    try:
        token = request.headers.get('Authorization')
        print("Token:", token)

        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
            print(actual_token)

            # Retrieve user_id and directory name from the database using token
            connection = get_database_connection()
            cursor = connection.cursor()
            cursor.execute('SELECT user_id FROM Customers WHERE api_token = %s', (actual_token,))
            user_id_result = cursor.fetchone()

            if user_id_result:
                user_id = user_id_result[0]
                print("User ID:", user_id)

                cursor.execute('SELECT directory_name FROM Directory WHERE user_id = %s', (user_id,))
                directory_info = cursor.fetchone()
                print(directory_info)

                if directory_info:
                    directory_name = directory_info[0]
                    print("Directory:", directory_name)

                    # Get the path of the saved file
                    file_path = os.path.join(BASE_UPLOAD_FOLDER, directory_name, "setting_pdf", "menu_structure.json")
                    print("File Path:", file_path)

                    if os.path.exists(file_path):
                        with open(file_path, 'r') as file:
                            file_content = json.load(file)
                        return jsonify(file_content)
                    else:
                        return jsonify({'error': 'File not found'}), 404
                else:
                    return jsonify({'error': 'Directory not found for the user'}), 404
            else:
                return jsonify({'error': 'User not found'}), 404
        else:
            return jsonify({'error': 'Invalid token'}), 401
    except Exception as e:
        send_error_email('get_file', str(e))
        return jsonify({'error': str(e)}), 500
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()
from flask import abort
from werkzeug.utils import safe_join


ZIP_DIRECTORY = "/home/plugin"
ZIP_FILE_NAME = "gentai_chatbot_plugin.zip"


@app.route('/wordpress_plugin', methods=['GET'])
def download_plugin():
    try:
        # Generate the full path to the file
        file_path = safe_join(ZIP_DIRECTORY, ZIP_FILE_NAME)
        app.logger.debug(f"File path: {file_path}")
        if os.path.isfile(file_path):
            return send_from_directory(ZIP_DIRECTORY, ZIP_FILE_NAME, as_attachment=True)
        else:
            app.logger.error(f"File not found: {file_path}")
            abort(404, description="Resource not found")
    except Exception as e:
        app.logger.error(f"Exception occurred: {str(e)}")
        abort(500, description=str(e))


















@app.route("/prompt", methods=["POST"])
def get_prompt_data():
    connection = None
    cursor = None
    try:
        connection = get_database_connection()
        cursor = connection.cursor()
        #time.sleep(10)

        data=request.get_json()
        # Extract parameters from the request URL
        SubmenuId = data['SubmenuId']
        Type = data['Type']
        Prompt = data['Prompt']
        PromptQuery=Prompt
        print(SubmenuId)
        # Extract token from request header
        token = request.headers.get('Authorization')
        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
        else:
            return jsonify({"error": "Invalid token format"})

        # Query the database to retrieve user_id associated with the token
        get_user_id_query = 'SELECT user_id FROM Customers WHERE user_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()
        if user_id_result:
            user_id = user_id_result[0]
            print(user_id)
        else:
            return jsonify({"error": "User not found for the provided token"})

        # Query the database based on the provided parameters and user_id
        query = """
            SELECT PromptValue,PromtId, PromtQuery, Score, SubmenuId, Type
            FROM QuestionData
            WHERE SubmenuId = %s AND PromtQuery = %s AND user_id = %s
        """
        cursor.execute(query, (SubmenuId,PromptQuery, user_id))
        prompt_data = cursor.fetchall()
        print(prompt_data)

        # Format the fetched data into the desired format
        formatted_data = []
        for row in prompt_data:
            formatted_data.append({
                "PromptValue": row[0],
                "PromtId": row[1],
                "PromtQuery": row[2],
                "Score": row[3],
                "SubmenuId": row[4],
                "Type": row[5]
            })

        return jsonify(formatted_data)

    except Exception as e:
        send_error_email('prompt', str(e))
        return jsonify({"error": str(e)})
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


def get_data_and_chroma_path_web_query(service_number):
    connection = get_database_connection()  # Assuming you have a function to establish a database connection
    cursor = connection.cursor()
    try:

        token = request.headers.get('Authorization')
        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
        else:
            actual_token = None
            response_data = {"message": "Invalid token format"}
            return None, None

        get_user_id_query = 'SELECT user_id FROM Customers WHERE user_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()

        if user_id_result:
            user_id = user_id_result[0]
            print(user_id)
        get_directory_query = 'SELECT directory_name FROM Directory WHERE user_id = %s'
        cursor.execute(get_directory_query, (user_id,))
        directory_info = cursor.fetchone()

        if directory_info:
            directory_name = directory_info[0]
            user_directory_path = os.path.join(BASE_UPLOAD_FOLDER, directory_name, f"service{service_number}")
            data_path = user_directory_path
            user_directory = os.path.join(BASE_UPLOAD_FOLDER, directory_name, f'service{service_number}_chromadb')
            chroma_path = user_directory
            print(chroma_path)
            return data_path, chroma_path
    except Exception as e:
        print("An error occurred:", e)
    finally:
        cursor.close()
        connection.close()
    return None, None

@app.route("/queries", methods=["POST"])
def query_chroma_web_queries():
    data = request.get_json()
    PromptId = data['PromptId']
    types = data['Type']
   # SubmenuId = data['SubmenuId']

    PROMPT_TEMPLATE = """
    Answer the question based only on the following context:
    GentAI, an AI PDF Expert, provides assistance based on the given context.
    To get started, please follow these steps:
    2. Describe the content.

    ---
    {context}
    ---
    Answer the question based on the above context: {question}
    """

    # Get data and chroma paths dynamically
    service_number = PromptId
    data_path, chroma_path = get_data_and_chroma_path_web_query(service_number)
    print(chroma_path)
    query_text = data['Question']
    print(query_text)
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)

    print(db)
    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)

    # Printing scores of each answer
    for doc, score in results:
        print("Score:", score)
        print("Document:", doc)  # Assuming you have a method to extract/document content
        print("---")

    if len(results) == 0 or results[0][1] < 0.7:
        # If no matching results found or the relevance score of the first result is less than 0.7
        # Handle the case where the answer is not confidently matched.
        model = ChatOpenAI()
        response_text = model.predict(query_text)
        response_text = "\n\nI am GentAI, the AI PDF Expert." + response_text

        sources = []  # No sources available since it's not based on context

        return jsonify({"response_text": response_text, "sources": sources})

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    print(context_text)
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt_kwargs = {
        "context": context_text,
        "question": query_text,
        "answer": ""  # Placeholder for the answer
    }
    prompt = prompt_template.format(**prompt_kwargs)

    model = ChatOpenAI()
    response_text = model.predict(prompt)
    print(response_text)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
   # formatted_response = [{"Answer": response_text,"sources":sources}]
    formatted_response = [{"Answer": response_text, "Score": score}]
    #formatted_response = [{"Answer": response_text, "Score": score, "sources": sources} for (doc, score), sources in zip(results, sources)]

    print(formatted_response)
    connection = get_database_connection()
    cursor = connection.cursor()
    try:

        token = request.headers.get('Authorization')
        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
            print("Inside_web" + actual_token)
        else:
            actual_token = None
            response_data = {"message": "Invalid token format"}
            return None, None
        get_user_id_query = 'SELECT user_id FROM Customers WHERE api_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()
        user_id = user_id_result[0]
        print(user_id)
        response_string = json.dumps(formatted_response)

        insert_chat_log_query = "INSERT INTO ChatLogs (user_id, question, answer,DateTimeColumn) VALUES (%s, %s, %s,CURRENT_TIMESTAMP)"
        chat_log_data = (user_id, query_text, response_string)
        print("inside+web", chat_log_data)
        cursor.execute(insert_chat_log_query, chat_log_data)
        connection.commit()

    except Exception as e:
  #      send_error_email('queries', str(e))
        print("An error occurred:", e)
    finally:
        cursor.close()
        connection.close()

    return jsonify(formatted_response)




@app.route('/get_pdf', methods=['GET'])
def get_pdf_file():
    connection = get_database_connection()
    cursor = connection.cursor()

    try:
        token = request.headers.get('Authorization')
        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
        else:
            return jsonify({'error': 'Invalid token'}), 401

        # Retrieve user_id based on the API token
        user_id_query = 'SELECT user_id FROM Customers WHERE api_token = %s'
        cursor.execute(user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()

        if user_id_result:
            user_id = user_id_result[0]

            # Retrieve directory name based on user_id
            directory_query = 'SELECT directory_name FROM Directory WHERE user_id = %s'
            cursor.execute(directory_query, (user_id,))
            directory_info = cursor.fetchone()

            if directory_info:
                directory_name = directory_info[0]
                pdf_directory = os.path.join(BASE_UPLOAD_FOLDER, directory_name, "pdfs")

                # Ensure the directory exists
                if not os.path.exists(pdf_directory):
                    return jsonify({'error': 'PDF directory not found'}), 404

                # Query the PDF file name from PDF_settings table
                pdf_settings_query = 'SELECT pdf_file1,pdf_file2,pdf_file3 FROM PDF_settings WHERE user_id = %s'
                cursor.execute(pdf_settings_query, (user_id,))
                pdf_filename = cursor.fetchone()
                if pdf_filenames:
                    # Assuming pdf_file1, pdf_file2, pdf_file3 are filenames
                    for pdf_filename in pdf_filenames:
                        if pdf_filename:
                            return send_from_directory(pdf_directory, pdf_filename)

                else:
                    return jsonify({'error': 'PDF file not found for the user'}), 404
            else:
                return jsonify({'error': 'Directory not found for the user'}), 404
        else:
            return jsonify({'error': 'User not found'}), 404
    finally:
        cursor.close()
        connection.close()



@app.route('/listing', methods=['POST'])
def listing():
    connection = get_database_connection()
    cursor = connection.cursor()

    token = request.headers.get('Authorization')

    try:
        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
        else:
            actual_token = None
            response_data = {"message": "Invalid token format"}
            return jsonify(response_data)

        get_user_id_query = 'SELECT user_id FROM Customers WHERE api_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()

        if user_id_result:
            user_id = user_id_result[0]

            get_directory_query = 'SELECT directory_name FROM Directory WHERE user_id = %s'
            cursor.execute(get_directory_query, (user_id,))
            directory_info = cursor.fetchone()

            if directory_info:
                directory_name = directory_info[0]
                user_directory_path = os.path.join(BASE_UPLOAD_FOLDER, directory_name, UPLOAD_FOLDER)
                DEFAULT_FOLDER_PATH = user_directory_path

                files = os.listdir(DEFAULT_FOLDER_PATH)

                cursor.close()
                connection.close()

                return jsonify({'files': files})
    except FileNotFoundError:
        return jsonify({'error': 'Folder not found'}), 404
    except mysql.connector.Error as err:
        return jsonify({'error': f'Database error: {err}'}), 500

    return jsonify({'error': 'Unknown error occurred'}), 500



from datetime import datetime, timedelta

import random
import string

@app.route('/days_registered', methods=['GET'])
def days_registered():
    connection = get_database_connection()
    cursor = connection.cursor()

    token = request.headers.get('Authorization')

    if not token or not token.startswith('Bearer '):
        return jsonify({'error': 'Unauthorized'}), 401

    try:
        actual_token = token.split('Bearer ')[1]

        user_query = 'SELECT user_id, registration_date, is_active FROM Customers WHERE api_token = %s'
        cursor.execute(user_query, (actual_token,))
        user_data = cursor.fetchone()

        if not user_data:
            return jsonify({'error': 'User not found'}), 404

        user_id, registration_date, is_active = user_data

        registration_date = registration_date.replace(tzinfo=None)  # Remove timezone info
        current_datetime = datetime.now()
        registration_time = current_datetime - registration_date

        # Check if registration time exceeds 1 minute
        if registration_time >= timedelta(days=200):
            # Disable the user in the database
            disable_user_query = 'UPDATE Customers SET is_active = 0, password = %s WHERE user_id = %s'
            random_password = ''.join(random.choices(string.ascii_letters + string.digits, k=12))  # Generate a random password
            cursor.execute(disable_user_query, (random_password, user_id,))
            connection.commit()  # Commit the changes to the database

            cursor.close()
            connection.close()

            return jsonify({'error': 'User disabled due to exceeding registration time limit', 'random_password': random_password}), 403

        days_registered = (current_datetime.date() - registration_date.date()).days

        cursor.close()
        connection.close()

        return jsonify({'user_id': user_id, 'days_registered': days_registered, 'is_active': is_active}), 200
    except mysql.connector.Error as err:
        send_error_email('days_registered', str(err))
        return jsonify({'error': f'Database error: {err}'}), 500



@app.route('/get_file_logs', methods=['GET'])
def get_filename_route():
    try:
        # Initialize the database connection and cursor
        connection = get_database_connection()
        cursor = connection.cursor()

        # Retrieve the Authorization token from the request headers
        token = request.headers.get('Authorization')

        # Check if the token is missing or not in the correct format
        if not token or not token.startswith('Bearer '):
            return jsonify({'error': 'Unauthorized'}), 401

        # Extract the actual token from the Authorization header
        actual_token = token.split('Bearer ')[1]

        # Query the database to get the user ID associated with the token
        user_query = 'SELECT user_id FROM Customers WHERE api_token = %s'
        cursor.execute(user_query, (actual_token,))
        user_id_result = cursor.fetchone()

        # Check if the user ID was found
        if not user_id_result:
            return jsonify({'error': 'User not found'}), 404

        user_id = user_id_result[0]

        # Query the database to get the filename and CreatedDateTimeColumn associated with the user ID
        file_query = 'SELECT file_name, CreatedDateTimeColumn FROM FileLogs WHERE user_id = %s'
        cursor.execute(file_query, (user_id,))
        filename_result = cursor.fetchall()

        # Check if a filename was found for the user
        if not filename_result:
            return jsonify({'error': 'File not found for this user'}), 404

        # Close cursor and connection
        cursor.close()
        connection.close()

        # Return the filename and CreatedDateTimeColumn as a JSON response
        return jsonify({'Files': [{'Filename': row[0], 'CreationDate': row[1]} for row in filename_result]}), 200
    except mysql.connector.Error as err:
        # Handle any database errors
        return jsonify({'error': f'Database error: {err}'}), 500
    except Exception as e:
        send_error_email('get_file_logs', str(e))
        # Handle any other unexpected errors
        return jsonify({'error': f'An error occurred: {e}'}), 500





def get_data_and_chroma_path_free_query(service_number):
    connection = get_database_connection()  # Assuming you have a function to establish a database connection
    cursor = connection.cursor()
    try:

        token = request.headers.get('Authorization')
        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
        else:
            actual_token = None
            response_data = {"message": "Invalid token format"}
            return None, None

        get_user_id_query = 'SELECT user_id FROM Customers WHERE user_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()

        if user_id_result:
            user_id = user_id_result[0]
            print(user_id)
        get_directory_query = 'SELECT directory_name FROM Directory WHERE user_id = %s'
        cursor.execute(get_directory_query, (user_id,))
        directory_info = cursor.fetchone()

        if directory_info:
            directory_name = directory_info[0]
            user_directory_path = os.path.join(BASE_UPLOAD_FOLDER, directory_name, f"service{service_number}")
            data_path = user_directory_path
            user_directory = os.path.join(BASE_UPLOAD_FOLDER, directory_name, f'service{service_number}_chromadb')
            chroma_path = user_directory
            print(chroma_path)
            return data_path, chroma_path
    except Exception as e:
        print("An error occurred:", e)
    finally:
        cursor.close()
        connection.close()
    return None, None
@app.route("/free_queries", methods=["POST"])
def query_chroma_free_queries():
    data = request.get_json()
    query_text = data['Question']

    # Define a list of service numbers
    service_numbers = ["1.1", "1.2", "1.3", "2.1", "2.2", "2.3", "3.1", "3.2", "3.3","4.1","4.2","4.3"]

    highest_score = -1  # Initialize highest score variable
    PROMPT_TEMPLATE = """
    Answer the question based only on the following context:
    Provides assistance based on the given context.
    To get started, please follow these steps:
    2. Describe the content.

    ---
    {context}
    ---
    Answer the question based on the above context: {question}
    """
    formatted_response = {"Answer": "No results found", "Score": -1}  # Default response if no results found

    for service_number in service_numbers:
        data_path, chroma_path = get_data_and_chroma_path_free_query(service_number)
        if data_path and chroma_path:  # Ensure paths are valid
            embedding_function = OpenAIEmbeddings()
            db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)
            results = db.similarity_search_with_relevance_scores(query_text, k=3)

            if results:  # Check if results are not empty
                if results[0][1] > highest_score:  # Update highest score and response if current score is higher
                    highest_score = results[0][1]
                    if highest_score < 0.7:
                        model = ChatOpenAI()
                        response_text = model.predict(query_text)
                        response_text = "\n\n" + response_text
                        sources = []  # No sources available since it's not based on context
                    else:
                        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
                        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
                        prompt_kwargs = {
                            "context": context_text,
                            "question": query_text,
                            "answer": ""  # Placeholder for the answer
                        }
                        prompt = prompt_template.format(**prompt_kwargs)
                        model = ChatOpenAI()
                        response_text = model.predict(prompt)
                        sources = [doc.metadata.get("source", None) for doc, _score in results]

                    formatted_response = {"Answer": response_text, "Score": highest_score}

    # Save chat logs outside the loop after processing all queries
    connection = get_database_connection()
    cursor = connection.cursor()
    try:
        token = request.headers.get('Authorization')
        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
        else:
            raise ValueError("Invalid token format")

        get_user_id_query = 'SELECT user_id FROM Customers WHERE api_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()
        user_id = user_id_result[0]

        response_string = json.dumps(formatted_response)
        insert_chat_log_query = "INSERT INTO ChatLogs (user_id, question, answer, DateTimeColumn) VALUES (%s, %s, %s, CURRENT_TIMESTAMP)"
        chat_log_data = (user_id, query_text, response_string)
        cursor.execute(insert_chat_log_query, chat_log_data)
        connection.commit()
    except Exception as e:
        send_error_email('free_queries', str(e))
        print("An error occurred:", e)
    finally:
        cursor.close()
        connection.close()

    return jsonify(formatted_response)


def get_data_and_chroma_path_web_queries(service_number):
    connection = get_database_connection()  # Assuming you have a function to establish a database connection
    cursor = connection.cursor()
    try:

        token = request.headers.get('Authorization')
        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
        else:
            actual_token = None
            response_data = {"message": "Invalid token format"}
            return None, None

        get_user_id_query = 'SELECT user_id FROM Customers WHERE user_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()
        print("serv=",service_number)

        if user_id_result:
            user_id = user_id_result[0]

        get_directory_query = 'SELECT directory_name FROM Directory WHERE user_id = %s'
        cursor.execute(get_directory_query, (user_id,))
        directory_info = cursor.fetchone()

        if directory_info:
            directory_name = directory_info[0]
            user_directory_path = os.path.join(BASE_UPLOAD_FOLDER, directory_name, f"service{service_number}")
            data_path = user_directory_path
            user_directory = os.path.join(BASE_UPLOAD_FOLDER, directory_name, f'service{service_number}_chromadb')
            chroma_path = user_directory
            print(chroma_path)
            return data_path, chroma_path
    except Exception as e:
        print("An error occurred:", e)
    finally:
        cursor.close()
        connection.close()
    return None, None












@app.route("/free_queries", methods=["POST"])
def query_chroma_web_free_queries():
    data = request.get_json()

    PROMPT_TEMPLATE = """
    # You will be acting as an AI PDF Expert named GentAI.
    # Your goal is to provide accurate answers and insights based on the given context.
    # You will be replying to users who may be confused if you don't respond appropriately.
    # You are provided with a PDF document for context.
    GentAI, an AI PDF Expert, provides assistance based on the given context.
    To get started, please follow these steps:
    1. Briefly introduce yourself as the AI PDF Expert GentAI.
    2. Describe the content.
    ---
    {context}
    ---
    Answer the question based on the above context: {question}
    """
    service_numbers = [1, 2, 3, 4]
    service_numbers = [float(str(num) + '.' + str(sub_num)) for num in combined_numbers for sub_num in range(1, 4)]

    all_responses = []  # To store responses for all service numbers

    # Get data and chroma paths dynamically
    for service_number in service_numbers:
        #service_number = PromptId
        print(service_number)
        data_path, chroma_path = get_data_and_chroma_path_web_query(service_number)
        print(chroma_path)
        query_text = data['Question']
        print(query_text)
        embedding_function = OpenAIEmbeddings()
        db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)

        print(db)
        # Search the DB.
        results = db.similarity_search_with_relevance_scores(query_text, k=3)

        # Printing scores of each answer
        for doc, score in results:
            print("Score:", score)
            print("Document:", doc)  # Assuming you have a method to extract/document content
            print("---")

        if len(results) == 0 or results[0][1] < 0.7:
            # If no matching results found or the relevance score of the first result is less than 0.7
            # Handle the case where the answer is not confidently matched.
            model = ChatOpenAI()
            response_text = model.predict(query_text)
            response_text = "\n\nI am GentAI, the AI PDF Expert." + response_text

            sources = []  # No sources available since it's not based on context
            return jsonify({"response_text": response_text, "sources": sources})

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        print(context_text)
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt_kwargs = {
            "context": context_text,
            "question": query_text,
            "answer": ""  # Placeholder for the answer
        }
        prompt = prompt_template.format(**prompt_kwargs)

        model = ChatOpenAI()
        response_text = model.predict(prompt)
        print(response_text)

        sources = [doc.metadata.get("source", None) for doc, _score in results]
        # formatted_response = [{"Answer": response_text,"sources":sources}]
        formatted_response = [{"Answer": response_text, "Score": score}]
        #formatted_response = [{"Answer": response_text, "Score": score, "sources": sources} for (doc, score), sources in zip(results, sources)]

        print(formatted_response)
        connection = get_database_connection()
        cursor = connection.cursor()
        try:

            token = request.headers.get('Authorization')
            if token and token.startswith('Bearer '):
                actual_token = token.split('Bearer ')[1]
                print("Inside_web" + actual_token)
            else:
                actual_token = None
                response_data = {"message": "Invalid token format"}
                return None, None
            get_user_id_query = 'SELECT user_id FROM Customers WHERE api_token = %s'
            cursor.execute(get_user_id_query, (actual_token,))
            user_id_result = cursor.fetchone()
            user_id = user_id_result[0]
            print(user_id)
            response_string = json.dumps(formatted_response)

            insert_chat_log_query = "INSERT INTO ChatLogs (user_id, question, answer,DateTimeColumn) VALUES (%s, %s, %s,CURRENT_TIMESTAMP)"
            chat_log_data = (user_id, query_text, response_string)
            print("inside+web", chat_log_data)
            cursor.execute(insert_chat_log_query, chat_log_data)
            connection.commit() 
        except Exception as e:
            send_error_email('free_queries', str(e))
            print("An error occurred:", e)
        finally:
            cursor.close()
            connection.close()

        return jsonify(formatted_response)

@app.route('/delete_log_files', methods=['POST'])
def delete_files():
    try:
        token = request.headers.get('Authorization')
        print("Token:", token)

        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
            print(actual_token)

            # Retrieve user_id and directory name from the database using token
            connection = get_database_connection()
            cursor = connection.cursor()
            cursor.execute('SELECT user_id FROM Customers WHERE api_token = %s', (actual_token,))
            user_id_result = cursor.fetchone()
            if user_id_result:
                user_id = user_id_result[0]
                print("User ID:", user_id)

                cursor.execute('SELECT directory_name FROM Directory WHERE user_id = %s', (user_id,))
                directory_info = cursor.fetchone()
                if directory_info:
                    directory_name = directory_info[0]
                    print("Directory:", directory_name)

                    # Get the filename to delete
                    data = request.json
                    print("Request Data:", data)
                    if 'filenames' in data:
                        filenames = data['filenames']
                        submenu_id = data.get('submenu_id')
                        print("Submenu ID:", submenu_id)
                        print("Filenames:", filenames)

                        for filename in filenames:
                            file_path = os.path.join(BASE_UPLOAD_FOLDER, directory_name, filename)
                            print("File Path:", file_path)
                            
                            # Check if file exists and delete it
                            if os.path.exists(file_path):
                                os.remove(file_path)
                                print("Deleted file:", file_path)

                        chroma_path = os.path.join(BASE_UPLOAD_FOLDER, directory_name, f"service{submenu_id}_chromadb")
                        if os.path.exists(chroma_path):
                            shutil.rmtree(chroma_path)
                            print("Deleted ChromaDB folder:", chroma_path)

                        # Remove entry from the database
                        upsert_file_query = '''
                            INSERT INTO reports (file_name, deleted, deleted_date, status, user_id)
                            VALUES (%s, 'Yes', CURRENT_TIMESTAMP, 'Disabled', %s)
                            ON DUPLICATE KEY UPDATE deleted = 'Yes', deleted_date = CURRENT_TIMESTAMP, status = 'Disabled'
                        '''
                        print("Filenames to delete:", filenames)
                        cursor.executemany(upsert_file_query, [(filename, user_id) for filename in filenames])

                        # Commit the database changes
                        connection.commit()

                        return jsonify({'message': 'Files deleted successfully'}), 200
                    else:
                        return jsonify({'error': 'Filename not provided in request'}), 400
                else:
                    return jsonify({'error': 'Directory not found'}), 404
            else:
                return jsonify({'error': 'User not found'}), 404
        else:
            return jsonify({'error': 'Invalid token'}), 401
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if cursor:
            cursor.close()



from langdetect import detect
from googletrans import Translator

translator = Translator()

def translate_text(text, target_language):
    translated_text = translator.translate(text, dest=target_language)
    return translated_text.text

def get_data_and_chroma_path():
    connection = get_database_connection()  # Assuming you have a function to establish a database connection
    cursor = connection.cursor()
    try:
        user_token = request.headers.get('Authorization')

        if user_token and user_token.startswith('Bearer '):
            user_token = user_token.split('Bearer ')[1]
            print(user_token)
        else:
            # Handle the case when the token is missing or in the wrong format
            return jsonify({'error': 'Invalid token'}), 401

        get_user_id_query = 'SELECT user_id FROM Customers WHERE user_token = %s'
        cursor.execute(get_user_id_query, (user_token,))
        user_id_result = cursor.fetchone()

        user_id = user_id_result[0]
        print(user_id)
        get_directory_query = 'SELECT directory_name FROM Directory WHERE user_id = %s'
        cursor.execute(get_directory_query, (user_id,))
        directory_info = cursor.fetchone()

        if directory_info:
            directory_name = directory_info[0]
            user_directory_path = os.path.join(BASE_UPLOAD_FOLDER, directory_name, UPLOAD_FOLDER)
            data_path = user_directory_path
            user_directory = os.path.join(BASE_UPLOAD_FOLDER, directory_name,'chroma')
            chroma_path = user_directory
            print(chroma_path)
            return data_path, chroma_path
    except Exception as e:
        print("An error occurred:", e)
    finally:
        cursor.close()
        connection.close()
    return None, None

@app.route("/query", methods=["POST"])
def query_chroma():
    PROMPT_TEMPLATE = """
    #To get started, please follow these steps:
    #1. Briefly introduce yourself as the AI PDF Expert GentAI.
    #2. Describe the content.
    #3. Provide 3 example questions using bullet points.

    #---

    # You will be acting as an AI PDF Expert named GentAI.
    # Your goal is to provide accurate answers and insights based on the given context.
    # You will be replying to users who may be confused if you don't respond appropriately.
    # You are provided with a PDF document for context.

    Answer the question based only on the following context:
    GentAI, an AI PDF Expert, provides assistance based on the given context.
    To get started, please follow these steps:
    1. Briefly introduce yourself as the AI PDF Expert GentAI.
    2. Describe the content.
    3. Provide 3 example questions using bullet points.

    ---

    {context}

    ---

    Answer the question based on the above context: {question}
    """

    data_path, chroma_path = get_data_and_chroma_path()
    data = request.json
    query_text = data['question']

    # Detect the language of the query
    detected_language = detect(query_text)
    
    # Translate the query to English if it's not already in English
    if detected_language != 'en':
        query_text = translate_text(query_text, 'en')
    
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)

    results = db.similarity_search_with_relevance_scores(query_text, k=3)

    if len(results) == 0 or results[0][1] < 0.7:
        model = ChatOpenAI()
        response_text = model.predict(query_text)
        response_text = "\n\nI am GentAI, the AI PDF Expert." + response_text

        sources = [] 

        return jsonify({"response_text": response_text, "sources": sources})

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt_kwargs = {
        "context": context_text,
        "question": query_text,
        "answer": ""
    }
    prompt = prompt_template.format(**prompt_kwargs)

    model = ChatOpenAI()
    response_text = model.predict(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    #formatted_response = {"response_text": response_text, "sources": sources}
    translated_response_text = translate_text(response_text, detected_language)
    formatted_response = {"response_text": translated_response_text, "sources": sources}
    connection = get_database_connection()  
    cursor = connection.cursor()
    try:
        token = request.headers.get('Authorization')
        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
            print("Inside_web"+actual_token)
        else:
            actual_token = None
            response_data = {"message": "Invalid token format"}
            return None, None
        get_user_id_query = 'SELECT user_id FROM Customers WHERE api_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()
        user_id=user_id_result[0]
        print(user_id)
        response_string = json.dumps(formatted_response)

        insert_chat_log_query = "INSERT INTO ChatLogs (user_id, question, answer,DateTimeColumn) VALUES (%s, %s, %s,CURRENT_TIMESTAMP)"
        chat_log_data = (user_id, query_text,response_string )
        print("inside+web",chat_log_data)
        cursor.execute(insert_chat_log_query, chat_log_data)
        connection.commit()

    except Exception as e:
        send_error_email('query', str(e))
        print("An error occurred:", e)
    finally:
        cursor.close()
        connection.close()

    return jsonify(formatted_response)



















@app.route('/get_cookies', methods=['GET'])
def get_cookies():
    connection = get_database_connection()
    cursor = connection.cursor()
    token = request.headers.get('Authorization')
    try:
        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
        else:
            return jsonify({'error': 'Invalid token'}), 401

        get_user_id_query = 'SELECT user_id FROM Customers WHERE api_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()

        if user_id_result:
            user_id = user_id_result[0]
            get_cookies_query = 'SELECT cookies_data, created_at FROM cookies WHERE user_id = %s'
            cursor.execute(get_cookies_query, (user_id,))
            cookies_data_result = cursor.fetchall()
            
            if cookies_data_result:
                cookies_list = []
                for row in cookies_data_result:
                    cookies_data = row[0]
                    created_at = row[1]
                    cookies_list.append({'Cookies': cookies_data, 'CreationDate': created_at})
                
                return jsonify(cookies_list), 200
            else:
                return jsonify({'message': 'No cookies found for the user'}), 404
        else:
            return jsonify({'error': 'User not found'}), 404

    except Exception as e:
        send_error_email('get_cookies', str(e))
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        connection.close()






@app.route('/get_url', methods=['GET'])
def get_url_route():
    connection = get_database_connection()
    cursor = connection.cursor()

    token = request.headers.get('Authorization')

    if not token or not token.startswith('Bearer '):
        return jsonify({'error': 'Unauthorized'}), 401

    actual_token = token.split('Bearer ')[1]

    try:
        user_query = 'SELECT user_id FROM Customers WHERE api_token = %s'
        cursor.execute(user_query, (actual_token,))
        user_id_result = cursor.fetchone()

        if not user_id_result:
            return jsonify({'error': 'User not found'}), 404

        user_id = user_id_result[0]

        file_url_query = 'SELECT url, log_date FROM WebURLlog WHERE user_id = %s'
        cursor.execute(file_url_query, (user_id,))
        url_result = cursor.fetchall()

        if not url_result:
            return jsonify({'message':'URL not found for the user'}),404

        urls_with_dates = [{'url': row[0], 'log_date': row[1]} for row in url_result]

        cursor.close()
        connection.close()

        return jsonify({'urls_with_dates': urls_with_dates}), 200

    except mysql.connector.Error as err:
        send_error_email('get_url', str(err))
        print("MySQL Error:", err)
        return jsonify({'Internal server error'})


@app.route('/ChatbotSettings', methods=['POST'])
def chat():
    connection = get_database_connection()
    cursor = connection.cursor()

    try:
        token = request.headers.get('Authorization')
        print("Token:", token)

        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
            print(actual_token)

            # Retrieve user_id from database using token
            cursor.execute('SELECT user_id FROM Customers WHERE api_token = %s', (actual_token,))
            user_id_result = cursor.fetchone()

            if user_id_result:
                user_id = user_id_result[0]
                print(user_id)
                # Extract data from request body
                data = request.json
                print(data)
                header = data.get('header')
                document_button = data.get('document_button')
                about_us_button = data.get('about_us_button')
                product_button = data.get('product_button')
                default_message = data.get('default_message')
                about_us_message = data.get('about_us_message')
            
                product_message= data.get('product_message')
                print(product_message)

                document_message = data.get('document_message')

                #Insert data into ChatbotSettings table
                #insert_query = '''
                 #   INSERT INTO ChatbotSettings
                  #  (header, Document_Button, About_us_Button, Product_Button, Default_message, About_us_message,Product_message, Document_message, user_id)
                   # VALUES
                    #(%s, %s, %s, %s, %s, %s, %s, %s,%s)
                #'''
                insert_query = '''
    INSERT INTO ChatbotSetting
    (header, Document_Button, About_us_Button, Product_Button, Default_message, About_us_message, Product_message, Document_message, user_id)
    VALUES
    (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
    header = VALUES(header),
    Document_Button = VALUES(Document_Button),
    About_us_Button = VALUES(About_us_Button),
    Product_Button = VALUES(Product_Button),
    Default_message = VALUES(Default_message),
    About_us_message = VALUES(About_us_message),
    Product_message = VALUES(Product_message),
    Document_message = VALUES(Document_message)
'''

                insert_data = (header, document_button, about_us_button, product_button, default_message, about_us_message,product_message,document_message, user_id)
                cursor.execute(insert_query, insert_data)
                print(insert_data)
                print(insert_query)
                connection.commit()

                return jsonify({'message': 'Data inserted successfully'}), 200
            else:
                return jsonify({'error': 'User not found'}), 404
        else:
            return jsonify({'error': 'Invalid token'}), 401
    except Exception as e:
        send_error_email('ChatbotSettings', str(e))
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        connection.close()







ALLOWED_EXTENSIONS = {'pdf'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

# Function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to get file size
def get_file_size(file):
    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)
    return size

# Function to handle file upload
def handle_file_upload(file, user_id, user_directory_path, cursor, connection):
    filename = secure_filename(file.filename)
    print(filename)
    path = os.path.join(user_directory_path, filename)
    file.save(path)
    insert_filename_query = 'INSERT INTO FileLogs (user_id, file_name) VALUES (%s, %s) ON DUPLICATE KEY UPDATE user_id = VALUES(user_id), file_name = VALUES(file_name)'
    cursor.execute(insert_filename_query, (user_id, filename))
    upsert_file_query = '''
        INSERT INTO reports (file_name, type, uploaded_date, status, user_id)
        VALUES (%s, 'file', CURRENT_TIMESTAMP, 'uploaded', %s)
        ON DUPLICATE KEY UPDATE uploaded_date = CURRENT_TIMESTAMP, status = 'uploaded'
    '''
    cursor.execute(upsert_file_query, (filename, user_id))
    connection.commit()  # Commit the transaction
    return filename

# Function to handle URL inclusion
def handle_url_inclusion(url, user_id, user_directory_path, cursor, connection):
    save_directory = user_directory_path
    print(save_directory)
    crawl_website(url, save_directory)
    insert_url_query = 'INSERT INTO WebURLlog (user_id,url) VALUES (%s, %s) ON DUPLICATE KEY UPDATE user_id = VALUES(user_id), url = VALUES(url)'
    cursor.execute(insert_url_query, (user_id, url))
    connection.commit()
    return url









@app.route('/logo_image_get/<filename>', methods=['GET'])
def get_image_logo(filename):
    try:
        token = request.headers.get('Authorization')
        print("Token:", token)

        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
            print(actual_token)

            # Retrieve user_id and directory name from the database using token
            connection = get_database_connection()
            cursor = connection.cursor()
            cursor.execute('SELECT user_id FROM Customers WHERE api_token = %s', (actual_token,))
            user_id_result = cursor.fetchone()

            if user_id_result:
                user_id = user_id_result[0]
                print("User ID:", user_id)

                cursor.execute('SELECT directory_name FROM Directory WHERE user_id = %s', (user_id,))
                directory_info = cursor.fetchone()
                print(directory_info)

                if directory_info:
                    directory_name = directory_info[0]
                    print("Directory:", directory_name)

                    # Check if the requested file exists in the specified directory
                    save_folder = os.path.join(BASE_UPLOAD_FOLDER, directory_name, "logo")
                    requested_file = os.path.join(save_folder, filename)

                    if os.path.exists(requested_file):
                        return send_from_directory(save_folder, filename)
                    else:
                        return jsonify({'error': 'File not found'}), 404
                else:
                    return jsonify({'error': 'Directory not found for the user'}), 404
            else:
                return jsonify({'error': 'User not found'}), 404
        else:
            return jsonify({'error': 'Invalid token'}), 401
    except Exception as e:
        send_error_email('logo_image_get/<filename>', str(e))
        return jsonify({'error': str(e)}), 500
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()











@app.route('/get_pdf_info', methods=['GET'])
def get_pdf_info():
    connection = get_database_connection()
    cursor = connection.cursor()
    try:
        token = request.headers.get('Authorization')
        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
        else:
            return jsonify({'error': 'Invalid token'}), 401

        get_user_id_query = 'SELECT user_id FROM Customers WHERE api_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()

        if user_id_result:
            user_id = user_id_result[0]
            cursor.execute('SELECT * FROM PDF_settings WHERE user_id = %s', (user_id,))
            pdf_settings = cursor.fetchone()
            print(pdf_settings)

            if pdf_settings:
                # Construct PDF settings data
                pdf_info = {
                    'pdf_file1': pdf_settings[2],
                    'pdf_file2': pdf_settings[3],
                    'pdf_file3': pdf_settings[4],
                    'title1': pdf_settings[5],
                    'title2': pdf_settings[6],
                    'title3': pdf_settings[7],
                    'description1': pdf_settings[8],
                    'description2': pdf_settings[9],
                    'description3': pdf_settings[10]
                }
                return jsonify(pdf_info)
            else:
                return jsonify({'error': 'PDF settings not found'}), 404
        else:
            return jsonify({'error': 'User not found'}), 404

    except Exception as e:
        send_error_email('get_pdf_info', str(e))
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        connection.close()


@app.route('/settingurls_get', methods=['GET'])
def settingurls_get():
    connection = None
    cursor = None
    try:
        connection = get_database_connection()
        cursor = connection.cursor()
        token = request.headers.get('Authorization')
        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
        else:
            return jsonify({'error': 'Invalid token'}), 401

        get_user_id_query = 'SELECT user_id FROM Customers WHERE api_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()

        if user_id_result:
            user_id = user_id_result[0]

            # Retrieve all URLs for the user
            get_urls_query = """
            SELECT url1, url2, url3, description1, description2, description3, Title1, Title2, Title3
            FROM setting_urls
            WHERE user_id = %s;
            """
            cursor.execute(get_urls_query, (user_id,))
            urls_result = cursor.fetchone()

            if urls_result:
                urls = {
                    'url1': urls_result[0],
                    'url2': urls_result[1],
                    'url3': urls_result[2],
                    'description1': urls_result[3],
                    'description2': urls_result[4],
                    'description3': urls_result[5],
                    'Title1': urls_result[6],
                    'Title2': urls_result[7],
                    'Title3': urls_result[8]
                }
                print(urls)
                return jsonify(urls), 200
            else:
                return jsonify({'message': 'No URLs found for the user'}), 404
        else:
            return jsonify({'error': 'User not found'}), 404

    except Exception as e:
        send_error_email('settingurls_get', str(e))
        return jsonify({'error': str(e)}), 500
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()




@app.route('/get_setting_images', methods=['GET'])
def get_setting_images():
    try:
        token = request.headers.get('Authorization')
        print("Token:", token)

        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
            print(actual_token)

            # Retrieve user_id and directory name from the database using token
            connection = get_database_connection()
            cursor = connection.cursor()
            cursor.execute('SELECT user_id FROM Customers WHERE api_token = %s', (actual_token,))
            user_id_result = cursor.fetchone()

            if user_id_result:
                user_id = user_id_result[0]
                print("User ID:", user_id)

                cursor.execute('SELECT directory_name FROM Directory WHERE user_id = %s', (user_id,))
                directory_info = cursor.fetchone()
                print(directory_info)

                if directory_info:
                    directory_name = directory_info[0]
                    print("Directory:", directory_name)

                    # Construct the path to the setting_image directory
                    setting_image_folder = os.path.join(BASE_UPLOAD_FOLDER, directory_name, "setting_image")

                    # Check if the directory exists
                    if os.path.exists(setting_image_folder):
                        # List all files in the directory
                        files = os.listdir(setting_image_folder)

                        # Return the list of file names as JSON response
                        return jsonify({'files': files})
                    else:
                        return jsonify({'error': 'Setting image directory not found'}), 404
                else:
                    return jsonify({'error': 'Directory not found for the user'}), 404
            else:
                return jsonify({'error': 'User not found'}), 404
        else:
            return jsonify({'error': 'Invalid token'}), 401
    except Exception as e:
        send_error_email('get_setting_images', str(e))
        return jsonify({'error': str(e)}), 500
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()



@app.route('/header_text_get', methods=['GET'])
def header_text_get():
    connection = None
    cursor = None

    try:
        # Get the authorization token from the request headers
        token = request.headers.get('Authorization')
        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]

        # Establish database connection
        connection = get_database_connection()
        cursor = connection.cursor()

        # Retrieve user ID using the provided token
        get_user_id_query = 'SELECT user_id FROM Customers WHERE api_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()

        if not user_id_result:
            return jsonify({'error': 'User not found or invalid token'}), 404

        user_id = user_id_result[0]

        # Retrieve header text for the user
        get_header_text_query = 'SELECT header_text1, header_text2 FROM header_text WHERE user_id = %s'
        cursor.execute(get_header_text_query, (user_id,))
        header_text_result = cursor.fetchone()

        if not header_text_result:
            return jsonify({'error': 'Header text not found for the user'}), 404

        header_text1, header_text2 = header_text_result

        return jsonify({
            'headerText1': header_text1,
            'headerText2': header_text2
        }), 200

    except mysql.connector.Error as err:
        return jsonify({'error': f"Database error: {err}"}), 500

    except Exception as e:
        send_error_email('header_text_get', str(e))
        return jsonify({'error': str(e)}), 500

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()




@app.route('/get_introduction_chatbot_message', methods=['GET'])
def introduction_chatbot_message_get():
    connection = None
    cursor = None

    try:
        # Get the authorization token from the request headers
        token = request.headers.get('Authorization')
        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]

        # Establish database connection
        connection = get_database_connection()
        cursor = connection.cursor()

        # Retrieve user ID using the provided token
        get_user_id_query = 'SELECT user_id FROM Customers WHERE api_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()

        if not user_id_result:
            return jsonify({'error': 'User not found or invalid token'}), 404

        user_id = user_id_result[0]

        # Retrieve introduction and descriptions for the user
        get_introduction_query = 'SELECT introduction, descriptions FROM introductions WHERE user_id = %s'
        cursor.execute(get_introduction_query, (user_id,))
        introduction_result = cursor.fetchone()

        if not introduction_result:
            return jsonify({'error': 'Introduction not found for the user'}), 404

        introduction, descriptions = introduction_result

        return jsonify({
            'introduction': introduction,
            'descriptions': descriptions
        }), 200

    except mysql.connector.Error as err:
        return jsonify({'error': f"Database error: {err}"}), 500

    except Exception as e:
        send_error_email('get_introduction_chatbot_message', str(e))
        return jsonify({'error': str(e)}), 500

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()







@app.route('/get_image_profile/<filename>', methods=['GET'])
def get_image_profile(filename):
    try:
        token = request.headers.get('Authorization')
        print("Token:", token)

        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
            print(actual_token)

            # Retrieve user_id and directory name from the database using token
            connection = get_database_connection()
            cursor = connection.cursor()
            cursor.execute('SELECT user_id FROM Customers WHERE api_token = %s', (actual_token,))
            user_id_result = cursor.fetchone()

            if user_id_result:
                user_id = user_id_result[0]
                print("User ID:", user_id)

                cursor.execute('SELECT directory_name FROM Directory WHERE user_id = %s', (user_id,))
                directory_info = cursor.fetchone()
                print(directory_info)

                if directory_info:
                    directory_name = directory_info[0]
                    print("Directory:", directory_name)

                    image_folder = os.path.join(BASE_UPLOAD_FOLDER, directory_name, "image")

                    # Check if the file exists
                    if os.path.exists(os.path.join(image_folder, filename)):
                        return send_from_directory(image_folder, filename)
                    else:
                        return jsonify({'error': 'Image not found'}), 404
                else:
                    return jsonify({'error': 'Directory not found for the user'}), 404
            else:
                return jsonify({'error': 'User not found'}), 404
        else:
            return jsonify({'error': 'Invalid token'}), 401
    except Exception as e:
        send_error_email('get_image_profile/<filename>', str(e))
        return jsonify({'error': str(e)}), 500
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


# API endpoint for file upload
@app.route("/include_file", methods=['POST'])
def upload_file():
    response_data = {"success": False, "message": ""}
    try:
        file = request.files.get('file')

        if file:
            # Check if file is allowed
            if not allowed_file(file.filename):
                response_data["message"] = "Only PDF files are allowed"
                return jsonify(response_data)

            # Check file size
            if get_file_size(file) > MAX_FILE_SIZE:
                response_data["message"] = "File size exceeds maximum limit (10 MB)"
                return jsonify(response_data)

            # Initialize cursor after establishing connection
            connection = get_database_connection()
            cursor = connection.cursor()

            token = request.headers.get('Authorization')

            # Check if the token starts with 'Bearer ' and extract the actual token
            if token and token.startswith('Bearer '):
                actual_token = token.split('Bearer ')[1]
            else:
                # Handle the case when the token is not in the expected format
                actual_token = None
                response_data["message"] = "Invalid token format"
                cursor.close()
                connection.close()
                return jsonify(response_data)

            # Retrieve the user's ID from the Customers table based on the token
            get_user_id_query = 'SELECT user_id, first_name FROM Customers WHERE api_token = %s'
            cursor.execute(get_user_id_query, (actual_token,))
            user_id_result = cursor.fetchone()

            if user_id_result:
                user_id, first_name = user_id_result
                user_folder_name = f"{first_name}_{user_id}"

                # Retrieve the user's directory name from the Directory table
                get_directory_query = 'SELECT directory_name FROM Directory WHERE user_id = %s'
                cursor.execute(get_directory_query, (user_id,))
                directory_info = cursor.fetchone()

                if directory_info:
                    directory_name = directory_info[0]
                    user_directory_path = os.path.join(BASE_UPLOAD_FOLDER, directory_name, UPLOAD_FOLDER)

                    # Handle file upload
                    filename = handle_file_upload(file, user_id, user_directory_path, cursor, connection)
                    print(filename['file'])

                    response_data["success"] = True
                    response_data["message"] = "File uploaded successfully"

                else:
                    response_data["message"] = "User directory not found"
            else:
                response_data["message"] = "Invalid token"

            cursor.close()
            connection.close()

    except Exception as e:
        send_error_email('include_file', str(e))
        print(f"Error: {e}")
        response_data["message"] = "An error occurred while processing the request"

    return jsonify(response_data)

# API endpoint for URL inclusion
@app.route("/include_url", methods=['POST'])
def include_url():
    response_data = {"success": False, "message": ""}
    try:
        include_url = request.form.get('include-url')

        if include_url:
            # Initialize cursor after establishing connection
            connection = get_database_connection()
            cursor = connection.cursor()

            token = request.headers.get('Authorization')

            # Check if the token starts with 'Bearer ' and extract the actual token
            if token and token.startswith('Bearer '):
                actual_token = token.split('Bearer ')[1]
            else:
                # Handle the case when the token is not in the expected format
                actual_token = None
                response_data["message"] = "Invalid token format"
                cursor.close()
                connection.close()
                return jsonify(response_data)

            # Retrieve the user's ID from the Customers table based on the token
            get_user_id_query = 'SELECT user_id, first_name FROM Customers WHERE api_token = %s'
            cursor.execute(get_user_id_query, (actual_token,))
            user_id_result = cursor.fetchone()

            if user_id_result:
                user_id, first_name = user_id_result
                user_folder_name = f"{first_name}_{user_id}"

                # Retrieve the user's directory name from the Directory table
                get_directory_query = 'SELECT directory_name FROM Directory WHERE user_id = %s'
                cursor.execute(get_directory_query, (user_id,))
                directory_info = cursor.fetchone()

                if directory_info:
                    directory_name = directory_info[0]
                    user_directory_path = os.path.join(BASE_UPLOAD_FOLDER, directory_name, UPLOAD_FOLDER)

                    # Handle URL inclusion
                    url = handle_url_inclusion(include_url, user_id, user_directory_path, cursor, connection)

                    response_data["success"] = True
                    response_data["message"] = "URL added successfully"

                else:
                    response_data["message"] = "User directory not found"
            else:
                response_data["message"] = "Invalid token"

            cursor.close()
            connection.close()

    except Exception as e:
        send_error_email('include_url', str(e))
        print(f"Error: {e}")
        response_data["message"] = "An error occurred while processing the request"

    return jsonify(response_data)











@app.route('/qna', methods=['POST'])
def qna():
    connection = get_database_connection()
    cursor = connection.cursor()

    try:
        token = request.headers.get('Authorization')
        print("Token:", token)

        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
            print(actual_token)

            # Retrieve user_id from database using token
            cursor.execute('SELECT user_id FROM Customers WHERE api_token = %s', (actual_token,))
            user_id_result = cursor.fetchone()

            if user_id_result:
                user_id = user_id_result[0]
                print(user_id)
                # Extract data from request body
                data = request.json
                print(data)
                
                # Extract additional questions and answers
                question_1 = data.get('question_1')
                question_2 = data.get('question_2')
                question_3 = data.get('question_3')
                answer_1 = data.get('answer_1')
                answer_2 = data.get('answer_2')
                answer_3 = data.get('answer_3')

                # Check if record exists for the given user_id
                cursor.execute('SELECT * FROM qna WHERE user_id = %s', (user_id,))
                existing_record = cursor.fetchone()

                if existing_record:
                    # Update the existing record
                    update_query = '''
                        UPDATE qna
                        SET question_1 = %s, question_2 = %s, question_3 = %s,
                            answer_1 = %s, answer_2 = %s, answer_3 = %s
                        WHERE user_id = %s
                    '''
                    update_data = (question_1, question_2, question_3, answer_1, answer_2, answer_3, user_id)
                    cursor.execute(update_query, update_data)
                    connection.commit()
                else:
                    # Insert a new record
                    insert_query = '''
                        INSERT INTO qna (user_id, question_1, question_2, question_3, answer_1, answer_2, answer_3)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    '''
                    insert_data = (user_id, question_1, question_2, question_3, answer_1, answer_2, answer_3)
                    cursor.execute(insert_query, insert_data)
                    connection.commit()

                return jsonify({'message': 'Data inserted or updated successfully'}), 200
            else:
                return jsonify({'error': 'User not found'}), 404
        else:
            return jsonify({'error': 'Invalid token'}), 401
    except Exception as e:
        send_error_email('qna', str(e))
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        connection.close()

@app.route('/get_qna', methods=['GET'])
def get_qna():
    connection = get_database_connection()
    cursor = connection.cursor()
    try:
        token = request.headers.get('Authorization')
        print("Token:", token)
        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
            print(actual_token)
            # Retrieve user_id from database using token
            cursor.execute('SELECT user_id FROM Customers WHERE api_token = %s', (actual_token,))
            user_id_result = cursor.fetchone()
            if user_id_result:
                user_id = user_id_result[0]
                print(user_id)
                # Retrieve Q&A data for the user
                cursor.execute('SELECT * FROM qna WHERE user_id = %s', (user_id,))
                qna_data = cursor.fetchone()
                print(qna_data)
                if qna_data:
                    qna_dict = {
                        'question_1': qna_data[2],
                        'answer_1': qna_data[5],
                        'question_2': qna_data[3],
                        'answer_2': qna_data[6],
                        'question_3': qna_data[4],
                        'answer_3': qna_data[7]

                    }
                    return jsonify(qna_dict), 200
                else:
                    return jsonify({'message': 'No Q&A data found for the user'}), 404
            else:
                return jsonify({'error': 'User not found'}), 404
        else:
            return jsonify({'error': 'Invalid token'}), 401
    except Exception as e:
        send_error_email('get_qna', str(e))
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        connection.close()


ALLOWED_EXTENSION = {'pdf'}
def allowed_files(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSION


@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    connection = get_database_connection()
    cursor = connection.cursor()

    try:
        token = request.headers.get('Authorization')
        print("Token:", token)

        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
            print(actual_token)

            # Retrieve user_id from database using token
            # Assuming get_database_connection() function is defined elsewhere
            connection = get_database_connection()
            cursor = connection.cursor()
            cursor.execute('SELECT user_id FROM Customers WHERE api_token = %s', (actual_token,))
            user_id_result = cursor.fetchone()

            if user_id_result:
                user_id = user_id_result[0]
                print("User ID:", user_id)

                # Check if PDF file is provided in the request
                if 'pdf' not in request.files:
                    return jsonify({'error': 'No PDF file provided'}), 400

                file = request.files['pdf']

                if file.filename == '':
                    return jsonify({'error': 'No selected file'}), 400

                if file and allowed_files(file.filename):
                    filename = secure_filename(file.filename)

                    # Retrieve directory name from the database based on user ID
                    cursor.execute('SELECT directory_name FROM Directory WHERE user_id = %s', (user_id,))
                    directory_info = cursor.fetchone()
                    print(directory_info)
                    if directory_info:
                        directory_name = directory_info[0]
                        user_directory_path = os.path.join(BASE_UPLOAD_FOLDER, directory_name, "setting_pdf")

                        # Create directory if it doesn't exist
                        if not os.path.exists(user_directory_path):
                            os.makedirs(user_directory_path)
                        print(user_directory_path)
                        file.save(os.path.join(user_directory_path, filename))

                        # Update database with the file path

                        return jsonify({'message': 'PDF file uploaded successfully'}), 200
                    else:
                        return jsonify({'error': 'Directory not found for the user'}), 404
                else:
                    return jsonify({'error': 'Invalid file format, only PDF files allowed'}), 400
            else:
                return jsonify({'error': 'User not found'}), 404
        else:
            return jsonify({'error': 'Invalid token'}), 401
    except Exception as e:
        send_error_email('upload_pdf', str(e))
        return jsonify({'error': str(e)}), 500
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


@app.route('/view_pdf/<filename>', methods=['GET'])
def view_pdf(filename):
    cursor = None
    connection = None
    try:
        token = request.headers.get('Authorization')
        print("Token:", token)

        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
            print(actual_token)

            # Retrieve user_id and directory name from the database using token
            connection = get_database_connection()
            cursor = connection.cursor()
            cursor.execute('SELECT user_id FROM Customers WHERE api_token = %s', (actual_token,))
            user_id_result = cursor.fetchone()
            user_id=user_id_result[0]
            print(user_id)

            cursor.execute('SELECT directory_name FROM Directory WHERE user_id = %s', (user_id,))
            directory_info = cursor.fetchone()
            print(directory_info)
            if directory_info:
                directory_name = directory_info[0]
                print("dir:", directory_name)



                # Construct the file path
                filepath = os.path.join(BASE_UPLOAD_FOLDER, directory_name, "setting_pdf", filename)

                if os.path.isfile(filepath):
                    return send_file(filepath, as_attachment=False)
                else:
                    abort(404)
            else:
                return jsonify({'error': 'User not found'}), 404
        else:
            return jsonify({'error': 'Invalid token'}), 401
    except Exception as e:
        send_error_email('view_pdf/<filename>', str(e))
        return jsonify({'error': str(e)}), 500
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

@app.route('/download_pdf/<filename>', methods=['GET'])
def download_pdf(filename):
    try:
        token = request.headers.get('Authorization')
        print("Token:", token)

        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
            print(actual_token)

            # Retrieve user_id and directory name from the database using token
            connection = get_database_connection()
            cursor = connection.cursor()
            cursor.execute('SELECT user_id FROM Customers WHERE api_token = %s', (actual_token,))
            user_id_result = cursor.fetchone()
            user_id=user_id_result[0]
            print(user_id)

            cursor.execute('SELECT directory_name FROM Directory WHERE user_id = %s', (user_id,))
            directory_info = cursor.fetchone()
            print(directory_info)
            if directory_info:
                directory_name = directory_info[0]
                print("dir:", directory_name)


                # Construct the file path
                filepath = os.path.join(BASE_UPLOAD_FOLDER, directory_name, "setting_pdf", filename)

                if os.path.isfile(filepath):
                    return send_file(filepath, as_attachment=True)
                else:
                    abort(404)
            else:
                return jsonify({'error': 'User not found'}), 404
        else:
            return jsonify({'error': 'Invalid token'}), 401
    except Exception as e:
        send_error_email('download_pdf/<filename>', str(e))
        return jsonify({'error': str(e)}), 500
    finally:
            cursor.close()
            connection.close()

@app.route('/view_pdf_list', methods=['GET'])
def view_pdf_list():
    try:
        token = request.headers.get('Authorization')
        print("Token:", token)

        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
            print(actual_token)

            # Retrieve user_id and directory name from the database using token
            connection = get_database_connection()
            cursor = connection.cursor()
            cursor.execute('SELECT user_id FROM Customers WHERE api_token = %s', (actual_token,))
            user_id_result = cursor.fetchone()
            user_id=user_id_result[0]
            print(user_id)

            cursor.execute('SELECT directory_name FROM Directory WHERE user_id = %s', (user_id,))
            directory_info = cursor.fetchone()
            print(directory_info)
            if directory_info:
                directory_name = directory_info[0]
                print("dir:", directory_name)

                # Fetch the list of PDFs from the specified directory
                pdf_directory = os.path.join(BASE_UPLOAD_FOLDER, directory_name, "setting_pdf")
                print(pdf_directory)
                pdf_files = [f for f in os.listdir(pdf_directory) if os.path.isfile(os.path.join(pdf_directory, f))]

                return jsonify({'pdfs': pdf_files})
            else:
                return jsonify({'error': 'User not found'}), 404
        else:
            return jsonify({'error': 'Invalid token'}), 401
    except Exception as e:
        send_error_email('view_pdf_list', str(e))
        
        return jsonify({'error': str(e)}), 500
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


@app.route('/get_chat_logs', methods=['GET'])
def get_chat_logs():
    connection = get_database_connection()
    cursor = connection.cursor()
    token = request.headers.get('Authorization')
    print("token=" + token)
    try:
        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
        else:
            # Handle the case when the token is missing or in the wrong format
            return jsonify({'error': 'Invalid token'}), 401

        # Retrieve user information using the token
        get_user_id_query = 'SELECT user_id FROM Customers WHERE api_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()

        if user_id_result:
            user_id = user_id_result[0]  # Extracting the user_id from the result

            # Fetch question, answer, and DateTimeColumn from ChatLogs table based on user_id
            get_chat_logs_query = 'SELECT question, answer, DateTimeColumn FROM ChatLogs WHERE user_id = %s'
            cursor.execute(get_chat_logs_query, (user_id,))
            chat_logs = cursor.fetchall()

            chat_logs_list = []
            for log in chat_logs:
                question, answer, datetime_column = log
                chat_logs_list.append({'Question': question, 'Answer': answer, 'CreationDate': datetime_column})

            return jsonify({'chat_logs': chat_logs_list}), 200
        else:
            return jsonify({'error': 'User not found'}), 404

    except Exception as e:
        send_error_email('get_chat_logs', str(e))
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        connection.close()


@app.route('/get_chatbot_settings', methods=['GET'])
def get_chatbot_settings():
    connection = get_database_connection()
    cursor = connection.cursor()
    token = request.headers.get('Authorization')
    print("token=" + token)
    try:
        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
            print(actual_token)
        else:
            # Handle the case when the token is missing or in the wrong format
            return jsonify({'error': 'Invalid token'}), 401

        # Retrieve user information using the token
        get_user_id_query = 'SELECT user_id FROM Customers WHERE api_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()

        if user_id_result:
            user_id = user_id_result[0]  # Extracting the user_id from the result
            print(user_id) 
            # Fetch required fields from ChatbotSettings table based on user_id
            get_settings_query = '''
                SELECT header, Document_Button, About_us_Button, Product_Button,
                       Default_message, About_us_message, Product_message, Document_message
                FROM ChatbotSetting
                WHERE user_id = %s
            '''
            cursor.execute(get_settings_query, (user_id,))
            settings_result = cursor.fetchone()
            print(settings_result)
            return jsonify({'chatbot_settings':settings_result}), 200
        else:
            return jsonify({'error': 'User not found'}), 404

    except Exception as e:
        send_error_email('get_chatbot_settings', str(e))
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        connection.close()

from flask import Flask, send_file
from zipfile import ZipFile

@app.route('/download_zip', methods=['GET'])
def download_zip():
    try:
        # Get the authorization token from the request headers
        token = request.headers.get('Authorization')

        # Check if the token is present and starts with 'Bearer '
        if not token or not token.startswith('Bearer '):
            return jsonify({'error': 'Unauthorized'}), 401

        # Extract the actual token value
        actual_token = token.split('Bearer ')[1]

        # Simulate fetching user data from the database using the provided token
        # For demonstration purposes, I'm assuming user data is available in this dictionary
        user_data = {'user_id': 123, 'first_name': 'John'}

        # Check if user data exists for the provided token
        if user_data:
            user_id = user_data['user_id']
            first_name = user_data['first_name']

            # Define the name of the zip file based on user's first name and id
            user_zip_name = f"{first_name}_{user_id}.zip"

            # Directory containing files to be zipped
            directory_to_zip = '/home/clients/code'

            # List of files to include in the zip
            files_to_zip = ['chatbot-plugin.php', 'script.js', 'style.css', 'template-chatbot.php']

            # Path to save the generated zip file
            zip_file_path = os.path.join(directory_to_zip, user_zip_name)

            # Create a zip file
            with ZipFile(zip_file_path, 'w') as zipf:
                for file in files_to_zip:
                    file_path = os.path.join(directory_to_zip, file)
                    if os.path.exists(file_path):
                        base_name = os.path.basename(file_path)
                        if base_name in zipf.namelist():
                            # If the file exists, rename it by appending a number to its name
                            file_name, file_extension = os.path.splitext(base_name)
                            index = 1
                            while f"{file_name}_{index}{file_extension}" in zipf.namelist():
                                index += 1
                            base_name = f"{file_name}_{index}{file_extension}"
                        zipf.write(file_path, base_name)

            # Check if the zip file was created successfully
            if os.path.exists(zip_file_path):
                # Create a response object
                response = send_file(zip_file_path, as_attachment=True)

                # Set the filename in the Content-Disposition header
                response.headers['Content-Disposition'] = f'attachment; filename="{user_zip_name}"'

                return response
            else:
                return jsonify({'error': 'Failed to create the zip file.'}), 500
        else:
            return jsonify({'error': 'User not found'}), 404
    except Exception as e:
        send_error_email('download_zip', str(e))
        # Handle any unexpected errors and return an appropriate response
        return jsonify({'error': str(e)}), 500


import base64

@app.route('/upsert_chatbot_settings', methods=['POST'])
def upsert_chatbot_settings():
    connection = get_database_connection()
    cursor = connection.cursor()
    token = request.headers.get('Authorization')
    try:
        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
        else:
            return jsonify({'error': 'Invalid token'}), 401

        get_user_id_query = 'SELECT user_id FROM Customers WHERE api_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()

        if user_id_result:
            user_id = user_id_result[0]
            print(user_id)
            request_data = request.json
            logo_base64 = request_data.get('logo')
            image_base64 = request_data.get('image')
            headercolor = request_data.get('headercolor')
            footercolor = request_data.get('footercolor')
            bodycolor = request_data.get('bodycolor')
            headermessage = request_data.get('headermessage')
            print(request_data)
            logo_data = base64.b64decode(logo_base64) if logo_base64 else None
            image_data = base64.b64decode(image_base64) if image_base64 else None

            upsert_query = '''
                INSERT INTO chatbotsetting (user_id, logo, image, headercolor, footercolor, bodycolor, headermessage)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                logo = VALUES(logo),
                image = VALUES(image),
                headercolor = VALUES(headercolor),
                footercolor = VALUES(footercolor),
                bodycolor = VALUES(bodycolor),
                headermessage = VALUES(headermessage)
            '''
            cursor.execute(upsert_query, (user_id, logo_data, image_data, headercolor, footercolor, bodycolor, headermessage))
            connection.commit()

            return jsonify({'message': 'Chatbot settings updated successfully'}), 200
        else:
            return jsonify({'error': 'User not found'}), 404

    except Exception as e:
        send_error_email('upsert_chatbot_settings', str(e))
        
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        connection.close()
@app.route('/lavel_0_chatbot_settings', methods=['POST'])
def lavel_0_chatbot_settings():
    connection = get_database_connection()
    cursor = connection.cursor()
    token = request.headers.get('Authorization')
    try:
        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
            print(actual_token)
        else:
            return jsonify({'error': 'Invalid token'}), 401

        get_user_id_query = 'SELECT user_id FROM Customers WHERE api_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()

        if user_id_result:
            user_id = user_id_result[0]
            request_data = request.json
            AI_service_name = request_data.get('AI_service_name')
            service_1 = request_data.get('service_1')
            service_2 = request_data.get('service_2')
            service_3 = request_data.get('service_3')
            service_4 = request_data.get('service_4')
            service_5 = request_data.get('service_5')
            print(request_data)
            upsert_query = '''
                INSERT INTO AI_Services (user_id, AI_service_name, service_1, service_2, service_3, service_4, service_5)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                AI_service_name = VALUES(AI_service_name),
                service_1 = VALUES(service_1),
                service_2 = VALUES(service_2),
                service_3 = VALUES(service_3),
                service_4 = VALUES(service_4),
                service_5 = VALUES(service_5)
            '''
            cursor.execute(upsert_query, (user_id, AI_service_name, service_1, service_2, service_3, service_4, service_5))
            print(upsert_query)
            connection.commit()

            return jsonify({'message': 'AI service settings updated successfully'}), 200
        else:
            return jsonify({'error': 'User not found'}), 404

    except Exception as e:
        send_error_email('lavel_0_chatbot_settings', str(e))
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        connection.close()
@app.route('/lavel_1_chatbot_settings', methods=['POST'])
def lavel_1_chatbot_settings():
    connection = get_database_connection()
    cursor = connection.cursor()
    token = request.headers.get('Authorization')
    try:
        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
            print(actual_token)
        else:
            return jsonify({'error': 'Invalid token'}), 401

        get_user_id_query = 'SELECT user_id FROM Customers WHERE api_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()

        if user_id_result:
            user_id = user_id_result[0]
            request_data = request.json
            Data_service_name = request_data.get('Data_service_name')  # Corrected column name
            service_1 = request_data.get('service_1')
            service_2 = request_data.get('service_2')
            service_3 = request_data.get('service_3')
            service_4 = request_data.get('service_4')
            service_5 = request_data.get('service_5')
            print(request_data)
            upsert_query = '''
                INSERT INTO Data_Services (user_id, Data_service_name, service_1, service_2, service_3, service_4, service_5)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                Data_service_name = VALUES(Data_service_name),  # Corrected column name
                service_1 = VALUES(service_1),
                service_2 = VALUES(service_2),
                service_3 = VALUES(service_3),
                service_4 = VALUES(service_4),
                service_5 = VALUES(service_5)
            '''
            cursor.execute(upsert_query, (user_id, Data_service_name, service_1, service_2, service_3, service_4, service_5))
            print(upsert_query)
            connection.commit()

            return jsonify({'message': 'Data service settings updated successfully'}), 200
        else:
            return jsonify({'error': 'User not found'}), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        connection.close()
@app.route('/lavel_2_chatbot_settings', methods=['POST'])
def lavel_2_chatbot_settings():
    connection = get_database_connection()
    cursor = connection.cursor()
    token = request.headers.get('Authorization')
    try:
        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
            print(actual_token)
        else:
            return jsonify({'error': 'Invalid token'}), 401

        get_user_id_query = 'SELECT user_id FROM Customers WHERE api_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()

        if user_id_result:
            user_id = user_id_result[0]
            request_data = request.json
            AI_news = request_data.get('AI_news')
            news_1 = request_data.get('news_1')
            news_2 = request_data.get('news_2')
            news_3 = request_data.get('news_3')
            news_4 = request_data.get('news_4')
            news_5 = request_data.get('news_5')
            print(request_data)
            upsert_query = '''
                INSERT INTO News_Services (user_id, AI_news, news_1, news_2, news_3, news_4, news_5)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                AI_news = VALUES(AI_news),
                news_1 = VALUES(news_1),
                news_2 = VALUES(news_2),
                news_3 = VALUES(news_3),
                news_4 = VALUES(news_4),
                news_5 = VALUES(news_5)
            '''
            cursor.execute(upsert_query, (user_id,AI_news, news_1, news_2, news_3, news_4, news_5))
            print(upsert_query)
            connection.commit()

            return jsonify({'message': 'News service settings updated successfully'}), 200
        else:
            return jsonify({'error': 'User not found'}), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        connection.close()
@app.route('/lavel_3_chatbot_settings', methods=['POST'])
def lavel_3_chatbot_settings():
    connection = get_database_connection()
    cursor = connection.cursor()
    token = request.headers.get('Authorization')
    try:
        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
        else:
            return jsonify({'error': 'Invalid token'}), 401

        get_user_id_query = 'SELECT user_id FROM Customers WHERE api_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()

        if user_id_result:
            user_id = user_id_result[0]
            request_data = request.json
            product_name = request_data.get('product_name')
            setup_calendaly = request_data.get('setup_calendaly')
            call_live_customer_support = request_data.get('call_live_customer_support')
            video_link = request_data.get('video_link')
            documentation_link = request_data.get('documentation_link')
            benefits = request_data.get('benefits')
            cost = request_data.get('cost')

            # Upsert query to insert or update data in the Product table
            upsert_query = '''
                INSERT INTO Product (user_id, product_name, setup_calendaly, call_live_customer_support, video_link, documentation_link, benefits, cost)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                product_name = VALUES(product_name),
                setup_calendaly = VALUES(setup_calendaly),
                call_live_customer_support = VALUES(call_live_customer_support),
                video_link = VALUES(video_link),
                documentation_link = VALUES(documentation_link),
                benefits = VALUES(benefits),
                cost = VALUES(cost)
            '''
            cursor.execute(upsert_query, (user_id, product_name, setup_calendaly, call_live_customer_support, video_link, documentation_link, benefits, cost))
            connection.commit()

            return jsonify({'message': 'Product settings updated successfully'}), 200
        else:
            return jsonify({'error': 'User not found'}), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        connection.close()

@app.route('/level_chatbot_settings', methods=['POST'])
def level_chatbot_settings():
    connection = get_database_connection()
    cursor = connection.cursor()
    token = request.headers.get('Authorization')
    try:
        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
            print(actual_token)
        else:
            return jsonify({'error': 'Invalid token'}), 401

        get_user_id_query = 'SELECT user_id FROM Customers WHERE api_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()

        if user_id_result:
            user_id = user_id_result[0]
            request_data = request.json
            setting_0 = request_data.get('setting_0')
            setting_1 = request_data.get('setting_1')
            setting_2 = request_data.get('setting_2')
            setting_3 = request_data.get('setting_3')

            upsert_query = '''
                INSERT INTO Settings (user_id, setting_0, setting_1, setting_2, setting_3)
                VALUES (%s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                setting_0 = VALUES(setting_0),
                setting_1 = VALUES(setting_1),
                setting_2 = VALUES(setting_2),
                setting_3 = VALUES(setting_3)
            '''
            cursor.execute(upsert_query, (user_id, setting_0, setting_1, setting_2, setting_3))
            print(upsert_query)
            connection.commit()

            return jsonify({'message': 'Settings updated successfully'}), 200
        else:
            return jsonify({'error': 'User not found'}), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        connection.close()



@app.route('/lavel_0_chatbot_service_settings', methods=['POST'])
def lavel_0_chatbot_service_settings():
    connection = get_database_connection()
    cursor = connection.cursor()
    token = request.headers.get('Authorization')
    try:
        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
        else:
            return jsonify({'error': 'Invalid token'}), 401
        print(actual_token)
        get_user_id_query = 'SELECT user_id FROM Customers WHERE api_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()
        print(user_id_result)

        if user_id_result:
            user_id = user_id_result[0]
            print(user_id)
            request_data = request.json
           # print(user_id)
            print(request_data)
            service_1 = request_data.get('service_1')
            service_2 = request_data.get('service_2')
            service_3 = request_data.get('service_3')
            service_4 = request_data.get('service_4')
            upsert_query = '''
                INSERT INTO Services (user_id, service_1, service_2, service_3, service_4)
                VALUES (%s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                service_1 = VALUES(service_1),
                service_2 = VALUES(service_2),
                service_3 = VALUES(service_3),
                service_4 = VALUES(service_4)
            '''
            print(upsert_query)
            cursor.execute(upsert_query, (user_id,service_1, service_2, service_3, service_4))
            connection.commit()

            return jsonify({'message': 'AI service settings updated successfully'}), 200
        else:
            return jsonify({'error': 'User not found'}), 404

    except Exception as e:
        send_error_email('lavel_0_chatbot_service_settings', str(e))
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        connection.close()






@app.route('/lavel_0_get_chatbot_settings', methods=['GET'])
def get_0_chatbot_settings():
    connection = get_database_connection()
    cursor = connection.cursor()
    token = request.headers.get('Authorization')

    try:
        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
        else:
            return jsonify({'error': 'Invalid token'}), 401

        get_user_id_query = 'SELECT user_id FROM Customers WHERE api_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()

        if user_id_result:
            user_id = user_id_result[0]
            select_query = '''
                SELECT AI_service_name, service_1, service_2, service_3, service_4, service_5
                FROM AI_Services
                WHERE user_id = %s
            '''
            cursor.execute(select_query, (user_id,))
            ai_service_settings = cursor.fetchone()

            if ai_service_settings:
                ai_settings_dict = {
                    'AI_service_name': ai_service_settings[0],
                    'service_1': ai_service_settings[1],
                    'service_2': ai_service_settings[2],
                    'service_3': ai_service_settings[3],
                    'service_4': ai_service_settings[4],
                    'service_5': ai_service_settings[5]
                }
                return jsonify(ai_settings_dict), 200
            else:
                return jsonify({'error': 'AI service settings not found'}), 404
        else:
            return jsonify({'error': 'User not found'}), 404

    except Exception as e:
        send_error_email('lavel_0_get_chatbot_settings', str(e))
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        connection.close()




@app.route('/level1_chatbot_settings', methods=['POST'])
def level1_chatbot_settings():
    connection = get_database_connection()
    cursor = connection.cursor()
    token = request.headers.get('Authorization')
    try:
        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
        else:
            return jsonify({'error': 'Invalid token'}), 401

        get_user_id_query = 'SELECT user_id FROM Customers WHERE api_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()

        if user_id_result:
            user_id = user_id_result[0]
            request_data = request.json
            setting_0 = request_data.get('setting_0')
            setting_1 = request_data.get('setting_1')
            setting_2 = request_data.get('setting_2')
            setting_3 = request_data.get('setting_3')

            upsert_query = '''
                INSERT INTO Settings (user_id, setting_0, setting_1, setting_2, setting_3)
                VALUES (%s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                setting_0 = VALUES(setting_0),
                setting_1 = VALUES(setting_1),
                setting_2 = VALUES(setting_2),
                setting_3 = VALUES(setting_3)
            '''
            cursor.execute(upsert_query, (user_id, setting_0, setting_1, setting_2, setting_3))
            connection.commit()

            return jsonify({'message': 'Settings updated successfully'}), 200
        else:
            return jsonify({'error': 'User not found'}), 404

    except Exception as e:
        send_error_email('level1_chatbot_settings', str(e))
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        connection.close()
@app.route('/save_cookies', methods=['POST'])
def save_cookies():
    connection = get_database_connection()
    cursor = connection.cursor()
    token = request.headers.get('Authorization')
    try:
        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
            print(actual_token)
        else:
            return jsonify({'error': 'Invalid token'}), 401

        get_user_id_query = 'SELECT user_id FROM Customers WHERE user_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()
        print(user_id_result)

        if user_id_result:
            user_id = user_id_result[0]
            print(user_id)
            cookies_data = request.json.get('cookies')
            print(cookies_data)
            insert_cookies_query = '''
                INSERT INTO cookies (user_id, cookies_data)
                VALUES (%s, %s)
            '''
            cursor.execute(insert_cookies_query, (user_id, cookies_data))
            connection.commit()

            return jsonify({'message': 'Cookies saved successfully'}), 200
        else:
            return jsonify({'error': 'User not found'}), 404

    except Exception as e:
        send_error_email('save_cookies', str(e))
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        connection.close()



from werkzeug.utils import secure_filename

@app.route('/save_file', methods=['POST'])
def save_file():
    try:
        token = request.headers.get('Authorization')
        print("Token:", token)

        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
            print(actual_token)

            # Retrieve user_id and directory name from the database using token
            connection = get_database_connection()
            cursor = connection.cursor()
            cursor.execute('SELECT user_id FROM Customers WHERE api_token = %s', (actual_token,))
            user_id_result = cursor.fetchone()

            if user_id_result:
                user_id = user_id_result[0]
                print("User ID:", user_id)

                cursor.execute('SELECT directory_name FROM Directory WHERE user_id = %s', (user_id,))
                directory_info = cursor.fetchone()
                print(directory_info)

                if directory_info:
                    directory_name = directory_info[0]
                    print("Directory:", directory_name)

                    # Save the received file to the specified directory, replacing any existing file
                    save_file = os.path.join(BASE_UPLOAD_FOLDER, directory_name, "setting_pdf")
                    print(save_file)
                    if not os.path.exists(save_file):
                        os.makedirs(save_file)
                    path=os.path.join(save_file,"menu_structure.json")
                    menu = request.json.get('menu')
                    print("Received file:",menu)  # Add debug print to check the received filename
                    with open(path, 'w') as file:
                        json.dump(menu, file)
                    #filename = secure_filename(file.filename)
                    #file.save(os.path.join(save_file, filename))

                    return jsonify({'message': 'File saved successfully'})
                else:
                    return jsonify({'error': 'Directory not found for the user'}), 404
            else:
                return jsonify({'error': 'User not found'}), 404
        else:
            return jsonify({'error': 'Invalid token'}), 401
    except Exception as e:
        send_error_email('save_file', str(e))
        return jsonify({'error': str(e)}), 500
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

@app.route('/get_menu', methods=['GET'])
def get_menu():
    try:
        token = request.headers.get('Authorization')
        print("Token:", token)

        if token and token.startswith('Bearer '):
            actual_token = token.split('Bearer ')[1]
            print(actual_token)

            # Retrieve user_id and directory name from the database using token
            connection = get_database_connection()
            cursor = connection.cursor()
            cursor.execute('SELECT user_id FROM Customers WHERE user_token = %s', (actual_token,))
            user_id_result = cursor.fetchone()

            if user_id_result:
                user_id = user_id_result[0]
                print("User ID:", user_id)

                cursor.execute('SELECT directory_name FROM Directory WHERE user_id = %s', (user_id,))
                directory_info = cursor.fetchone()
                print(directory_info)

                if directory_info:
                    directory_name = directory_info[0]
                    print("Directory:", directory_name)

                    file_path = os.path.join(BASE_UPLOAD_FOLDER, directory_name, "setting_pdf", "menu_structure.json")

                    if os.path.exists(file_path):
                        with open(file_path, 'r') as file:
                            menu_data = json.load(file)
                            return jsonify(menu_data)
                    else:
                        return jsonify({'error': 'File not found'}), 404
                else:
                    return jsonify({'error': 'Directory not found for the user'}), 404
            else:
                return jsonify({'error': 'User not found'}), 404
        else:
            return jsonify({'error': 'Invalid token'}), 401
    except Exception as e:
        send_error_email('get_menu', str(e))
        return jsonify({'error': str(e)}), 500
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


@app.route('/add_data', methods=['POST'])
def add_data():
    try:
        # Get the user's name from the request body
        user_name = request.json.get('userName')

        # Do something with the user's name, for example, print it
        print("User's name:", user_name)

        # You can perform other operations here

        # Return a JSON response
        return jsonify({"message": "Data added successfully"}), 200
    except Exception as e:
        send_error_email('add_data', str(e))
        return jsonify({"error": str(e)}), 500


