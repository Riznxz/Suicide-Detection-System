from flask import Flask, request, jsonify, render_template, redirect, url_for, session, send_from_directory
import pickle
import numpy as np
import os
import contextlib
import sqlite3
import re
import pickle
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
import uuid
from werkzeug.utils import secure_filename
from PIL import Image
from create_database import setup_database
from utils import login_required, set_session
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load trained model, vectorizer, and encoder
model = pickle.load(open('models/model.pkl', 'rb'))
vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))
encoder = pickle.load(open('models/encoder.pkl', 'rb'))
database = "users.db"
setup_database(name=database)



@app.route('/')
def home():
    return render_template('index1.html')
@app.route('/predict', methods=['GET'])
def predict_form():
    return render_template('index.html')

# Route to handle user logout
@app.route('/logout')
def logout():
    session.clear()
    session.permanent = False
    return redirect('/login')

# Route for user login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')

    # Get form data
    username = request.form.get('username')
    password = request.form.get('password')
    
    # Query the database for user credentials
    query = 'SELECT username, password, email FROM users WHERE username = :username'
    with contextlib.closing(sqlite3.connect(database)) as conn:
        with conn:
            account = conn.execute(query, {'username': username}).fetchone()

    if not account:
        return render_template('login.html', error='Username does not exist')

    # Verify password
    try:
        ph = PasswordHasher()
        ph.verify(account[1], password)
    except VerifyMismatchError:
        return render_template('login.html', error='Incorrect password')

    # Update password if needed
    if ph.check_needs_rehash(account[1]):
        query = 'UPDATE users SET password = :password WHERE username = :username'
        params = {'password': ph.hash(password), 'username': account[0]}
        with contextlib.closing(sqlite3.connect(database)) as conn:
            with conn:
                conn.execute(query, params)

    # Set user session
    set_session(username=account[0], email=account[2], remember_me='remember-me' in request.form)
    return redirect('/predict')

# Route for user registration
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('register.html')

    # Get form data
    password = request.form.get('password')
    confirm_password = request.form.get('confirm-password')
    username = request.form.get('username')
    email = request.form.get('email')

    # Validate registration data
    if len(password) < 8:
        return render_template('register.html', error='Password must be at least 8 characters')
    if password != confirm_password:
        return render_template('register.html', error='Passwords do not match')
    if not re.match(r'^[a-zA-Z0-9]+$', username):
        return render_template('register.html', error='Username must only contain letters and numbers')
    if not 3 < len(username) < 26:
        return render_template('register.html', error='Username must be between 4 and 25 characters')

    # Check if username already exists
    query = 'SELECT username FROM users WHERE username = :username'
    with contextlib.closing(sqlite3.connect(database)) as conn:
        with conn:
            result = conn.execute(query, {'username': username}).fetchone()

    if result:
        return render_template('register.html', error='Username already exists')

    # Hash the password and save the user in the database
    pw = PasswordHasher()
    hashed_password = pw.hash(password)
    query = 'INSERT INTO users (username, password, email) VALUES (:username, :password, :email)'
    params = {'username': username, 'password': hashed_password, 'email': email}
    with contextlib.closing(sqlite3.connect(database)) as conn:
        with conn:
            conn.execute(query, params)

    # Log the user in immediately
    set_session(username=username, email=email)
    return redirect('/')


@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    transformed_text = vectorizer.transform([text]).toarray()
    prediction = model.predict(transformed_text)[0]
    
    label = encoder.inverse_transform([prediction])[0]
    return render_template('result.html', prediction=label)

if __name__ == '__main__':
    app.run(debug=True)
