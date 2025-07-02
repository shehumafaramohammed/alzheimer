# from flask import Flask, request, render_template, jsonify, redirect, url_for, flash, session
# import pandas as pd
# import numpy as np
# import joblib
# import json
# import os
# import sys
# import traceback
# import tensorflow as tf
# from werkzeug.utils import secure_filename
# from werkzeug.security import generate_password_hash, check_password_hash
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from flask_sqlalchemy import SQLAlchemy
# from datetime import datetime

# # Print version info for debugging
# print(f"Python version: {sys.version}")
# print(f"NumPy version: {np.__version__}")
# print(f"Pandas version: {pd.__version__}")
# try:
#     import sklearn

#     print(f"scikit-learn version: {sklearn.__version__}")
# except ImportError:
#     print("scikit-learn not installed")

# # Create application directory structure if it doesn't exist
# for directory in ['templates', 'static', 'static/uploads', 'instance']:
#     if not os.path.exists(directory):
#         os.makedirs(directory)
#         print(f"Created {directory} directory")

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'static/uploads'
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
# app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# app.secret_key = 'your_secret_key_here'  # Change this to a random secret key in production

# # Initialize SQLAlchemy
# db = SQLAlchemy(app)


# # User model
# class User(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     username = db.Column(db.String(80), unique=True, nullable=False)
#     email = db.Column(db.String(120), unique=True, nullable=False)
#     password_hash = db.Column(db.String(256), nullable=False)
#     created_at = db.Column(db.DateTime, default=datetime.utcnow)

#     def __repr__(self):
#         return f'<User {self.username}>'


# # Create the database tables
# with app.app_context():
#     db.create_all()
#     print("Database tables created")

# # Global variables for the traditional ML model
# ml_model = None
# scaler = None
# feature_columns = []  # Will be populated from feature_names.json
# original_features = []  # Will be populated from feature_names.json
# encoded_features = []  # Will be populated from feature_names.json
# numerical_columns = []  # Will be populated from feature_names.json
# categorical_columns = []  # Will be populated from feature_names.json
# diagnosis_mapping = {0: 'Non-Demented', 1: 'Mild Demented', 2: 'Moderate Demented', 3: 'Very Mild Demented'}

# # Global variables for the image-based model
# image_model = None
# class_labels = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']


# def allowed_file(filename):
#     """Check if the file has an allowed extension"""
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# def load_models():
#     """Load all models and their dependencies"""
#     global ml_model, scaler, original_features, encoded_features, numerical_columns, categorical_columns, diagnosis_mapping, image_model, feature_columns

#     # Try to load the traditional ML model and associated files
#     try:
#         print("Loading traditional ML model from Alzheimer_detection.pkl...")
#         ml_model = joblib.load('Alzheimer_detection.pkl')
#         print("ML model loaded successfully!")

#         print("Loading scaler from scaler.pkl...")
#         scaler = joblib.load('scaler.pkl')
#         print("Scaler loaded successfully!")

#         # Try to load feature information
#         try:
#             print("Loading feature information from feature_names.json...")
#             with open('feature_names.json', 'r') as f:
#                 feature_info = json.load(f)
#                 original_features = feature_info.get('original_features', [])
#                 encoded_features = feature_info.get('encoded_features', [])
#                 numerical_columns = feature_info.get('numerical_columns', [])
#                 categorical_columns = feature_info.get('categorical_columns', [])

#             # Set feature columns for the form
#             feature_columns = original_features
#             print(f"Loaded feature information:")
#             print(f"Original features ({len(original_features)}): {original_features}")
#             print(f"Encoded features ({len(encoded_features)}): {encoded_features}")
#             print(f"Numerical columns ({len(numerical_columns)}): {numerical_columns}")
#             print(f"Categorical columns ({len(categorical_columns)}): {categorical_columns}")

#         except FileNotFoundError:
#             print("Warning: feature_names.json not found. Using default feature columns.")
#             # Set some reasonable defaults if feature_names.json doesn't exist
#             feature_columns = ['Age', 'Gender', 'MMSE', 'CDR']
#             original_features = feature_columns.copy()
#             encoded_features = feature_columns.copy()
#             numerical_columns = feature_columns.copy()

#         # Try to load class mapping
#         try:
#             print("Loading class labels from class_labels.json...")
#             with open('class_labels.json', 'r') as f:
#                 class_mapping = json.load(f)
#                 # Convert string keys back to integers
#                 diagnosis_mapping = {int(k): v for k, v in class_mapping.items()}
#             print(f"Loaded {len(diagnosis_mapping)} class labels")

#         except FileNotFoundError:
#             print("Warning: class_labels.json not found. Using default diagnosis mapping.")

#     except Exception as e:
#         print(f"Error loading ML model: {str(e)}")
#         traceback.print_exc()
#         print("WARNING: The ML-based prediction functionality won't work without the model")

#     # Try to load the image-based model
#     try:
#         print("Loading image-based model from improved_alzheimers_model.keras...")
#         image_model = tf.keras.models.load_model("improved_alzheimers_model.keras")
#         print("Image model loaded successfully!")
#     except Exception as e:
#         print(f"Error loading image model: {str(e)}")
#         traceback.print_exc()
#         print("WARNING: The image-based prediction functionality won't work without the model")


# def preprocess_image(image_path):
#     """Preprocess an image for prediction."""
#     IMG_SIZE = (256, 256)
#     try:
#         img = load_img(image_path, target_size=IMG_SIZE)
#         img_array = img_to_array(img) / 255.0  # Normalize
#         img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
#         return img_array
#     except Exception as e:
#         print(f"Error preprocessing image: {str(e)}")
#         traceback.print_exc()
#         return None


# def predict_alzheimers_from_image(image_path):
#     """Predict Alzheimer's status from an image."""
#     if image_model is None:
#         return "Model not loaded", 0

#     img_array = preprocess_image(image_path)
#     if img_array is None:
#         return "Error processing image", 0

#     prediction = image_model.predict(img_array)
#     predicted_class = np.argmax(prediction, axis=1)[0]
#     confidence = float(np.max(prediction) * 100)

#     return class_labels[predicted_class], confidence


# # User authentication helper
# def login_required(view_func):
#     """Decorator to require login for certain views"""

#     def wrapped_view(*args, **kwargs):
#         if 'user_id' not in session:
#             flash('Please log in to access this page')
#             return redirect(url_for('login'))
#         return view_func(*args, **kwargs)

#     wrapped_view.__name__ = view_func.__name__
#     return wrapped_view


# # Authentication routes
# @app.route('/register', methods=['GET', 'POST'])
# def register():
#     """User registration page"""
#     if request.method == 'POST':
#         username = request.form.get('username')
#         email = request.form.get('email')
#         password = request.form.get('password')
#         confirm_password = request.form.get('confirm_password')

#         # Validate form data
#         if not username or not email or not password:
#             flash('All fields are required')
#             return render_template('register.html')

#         if password != confirm_password:
#             flash('Passwords must match')
#             return render_template('register.html')

#         # Check if username or email already exists
#         existing_user = User.query.filter((User.username == username) | (User.email == email)).first()
#         if existing_user:
#             flash('Username or email already in use')
#             return render_template('register.html')

#         # Create new user
#         hashed_password = generate_password_hash(password)
#         new_user = User(username=username, email=email, password_hash=hashed_password)
#         db.session.add(new_user)

#         try:
#             db.session.commit()
#             flash('Registration successful! Please log in.')
#             return redirect(url_for('login'))
#         except Exception as e:
#             db.session.rollback()
#             flash(f'Registration failed: {str(e)}')
#             return render_template('register.html')

#     return render_template('register.html')


# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     """User login page"""
#     if request.method == 'POST':
#         username = request.form.get('username')
#         password = request.form.get('password')

#         # Find user by username
#         user = User.query.filter_by(username=username).first()

#         if user and check_password_hash(user.password_hash, password):
#             session['user_id'] = user.id
#             session['username'] = user.username
#             flash('Login successful!')
#             return redirect(url_for('dashboard'))
#         else:
#             flash('Invalid username or password')

#     return render_template('login.html')


# @app.route('/logout')
# def logout():
#     """Log out a user"""
#     session.pop('user_id', None)
#     session.pop('username', None)
#     flash('You have been logged out')
#     return redirect(url_for('home'))


# @app.route('/dashboard')
# @login_required
# def dashboard():
#     """User dashboard"""
#     return render_template('dashboard.html')


# # Main routes
# @app.route('/')
# def home():
#     """Render the home page with options for both detection methods"""
#     return render_template('index.html')


# @app.route('/form')
# @login_required
# def form():
#     """Render the traditional ML model form"""
#     # Check if feature information is loaded
#     if not feature_columns:
#         flash('Warning: No feature information available. Using default features.')

#     return render_template('form.html',
#                            feature_columns=feature_columns,
#                            numerical_columns=numerical_columns,
#                            categorical_columns=categorical_columns)


# @app.route('/upload')
# @login_required
# def upload():
#     """Render the image upload form"""
#     return render_template('upload.html')


# @app.route('/predict_ml', methods=['POST'])
# @login_required
# def predict_ml():
#     """Endpoint to make predictions based on form input (traditional ML model)"""
#     if request.method == 'POST':
#         try:
#             # Check if model and scaler are loaded
#             if ml_model is None or scaler is None:
#                 return render_template('error.html',
#                                        error="ML model or scaler not loaded. Please check server logs and make sure both files exist.")

#             # Get values from form
#             features = {}
#             print("Processing form inputs:")

#             # Process all expected original features
#             for column in original_features:
#                 value = request.form.get(column)
#                 print(f"  - {column}: {value}")

#                 if value is not None and value != '':
#                     # Try to convert numerical values
#                     if column in numerical_columns:
#                         try:
#                             features[column] = float(value)
#                         except ValueError:
#                             # Fallback for invalid numerical values
#                             features[column] = 0
#                             print(f"Warning: Invalid numerical value for {column}, using 0")
#                     else:
#                         # For categorical values, keep as string
#                         features[column] = value
#                 else:
#                     # Handle missing values
#                     if column in numerical_columns:
#                         features[column] = 0  # Default for numerical
#                     else:
#                         features[column] = ""  # Default for categorical
#                     print(f"Warning: Missing value for {column}, using default")

#             # Convert to DataFrame
#             input_df = pd.DataFrame([features])
#             print(f"Input dataframe created with shape: {input_df.shape}")

#             # Handle categorical columns with one-hot encoding
#             if categorical_columns:
#                 print(f"Applying one-hot encoding to categorical columns: {categorical_columns}")
#                 input_df = pd.get_dummies(input_df, columns=categorical_columns)
#                 print(f"After one-hot encoding, shape: {input_df.shape}")

#             # Get expected features for model
#             model_columns = encoded_features
#             print(f"Model expects {len(model_columns)} features: {model_columns}")

#             # For compatibility with older scikit-learn versions
#             # Ensure all expected columns exist
#             for col in model_columns:
#                 if col not in input_df.columns:
#                     print(f"Adding missing column: {col}")
#                     input_df[col] = 0

#             # Remove extra columns not expected by model
#             for col in list(input_df.columns):
#                 if col not in model_columns:
#                     print(f"Removing extra column: {col}")
#                     input_df = input_df.drop(col, axis=1)

#             # Ensure dataframe has all columns in the right order
#             input_df = input_df[model_columns]
#             print(f"Final input dataframe shape: {input_df.shape}")
#             print(f"Final columns: {input_df.columns.tolist()}")

#             # Apply scaling
#             print("Applying scaling transformation...")
#             input_array = scaler.transform(input_df)

#             # Make prediction
#             print("Making prediction...")
#             prediction = ml_model.predict(input_array)
#             print(f"Raw prediction value: {prediction[0]}")

#             # Calculate confidence score if possible
#             try:
#                 prediction_proba = ml_model.predict_proba(input_array)
#                 confidence = np.max(prediction_proba) * 100
#                 print(f"Confidence: {confidence:.2f}%")
#             except:
#                 print("Could not calculate confidence (predict_proba not available)")
#                 confidence = None

#             # Map prediction to diagnosis label
#             diagnosis = diagnosis_mapping.get(int(prediction[0]), f"Unknown Class ({prediction[0]})")
#             print(f"Mapped diagnosis: {diagnosis}")

#             # Return the result
#             return render_template('result.html',
#                                    diagnosis=diagnosis,
#                                    confidence=confidence,
#                                    features=features,
#                                    prediction_type="Feature-based")

#         except Exception as e:
#             print(f"Error during ML prediction: {str(e)}")
#             traceback.print_exc()
#             return render_template('error.html',
#                                    error=f"An error occurred during prediction: {str(e)}")

#     return redirect(url_for('form'))


# @app.route('/predict_image', methods=['POST'])
# @login_required
# def predict_image():
#     """Endpoint to make predictions based on uploaded image"""
#     if request.method == 'POST':
#         try:
#             # Check if model is loaded
#             if image_model is None:
#                 return render_template('error.html',
#                                        error="Image model not loaded. Please check server logs and make sure the model file exists.")

#             # Check if a file was uploaded
#             if 'file' not in request.files:
#                 return render_template('error.html', error="No file part")

#             file = request.files['file']

#             # Check if a file was selected
#             if file.filename == '':
#                 return render_template('error.html', error="No selected file")

#             # Check if the file is allowed
#             if file and allowed_file(file.filename):
#                 filename = secure_filename(file.filename)
#                 filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#                 file.save(filepath)

#                 # Make prediction
#                 diagnosis, confidence = predict_alzheimers_from_image(filepath)

#                 # Return the result
#                 return render_template('image_result.html',
#                                        diagnosis=diagnosis,
#                                        confidence=confidence,
#                                        image_path=filepath,
#                                        prediction_type="Image-based")
#             else:
#                 return render_template('error.html',
#                                        error="File type not allowed. Please upload a JPG, JPEG, or PNG image.")

#         except Exception as e:
#             print(f"Error during image prediction: {str(e)}")
#             traceback.print_exc()
#             return render_template('error.html',
#                                    error=f"An error occurred during prediction: {str(e)}")

#     return redirect(url_for('upload'))


# # Helper context processor for templates
# @app.context_processor
# def utility_processor():
#     def now():
#         return datetime.now()

#     return dict(now=now)


# if __name__ == '__main__':
#     load_models()
#     print(f"Starting Flask app on http://127.0.0.1:5000/")
#     print(f"Press CTRL+C to quit")
#     app.run(debug=True)
# from flask import Flask, request, render_template, jsonify, redirect, url_for, flash, session
# import pandas as pd
# import numpy as np
# import joblib
# import json
# import os
# import sys
# import traceback
# import tensorflow as tf
# import requests
# from werkzeug.utils import secure_filename
# from werkzeug.security import generate_password_hash, check_password_hash
# # EDIT: This is the corrected import path for modern TensorFlow versions
# from tensorflow.keras.utils import load_img, img_to_array
# from flask_sqlalchemy import SQLAlchemy
# from datetime import datetime

# # --- CONFIGURATION ---
# OPENROUTER_API_KEY = "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" # It's better to use environment variables for keys
# FLASK_SECRET_KEY = 'a-very-strong-and-random-secret-key-that-you-should-change'

# # --- APP SETUP ---
# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'static/uploads'
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# app.secret_key = FLASK_SECRET_KEY
# db = SQLAlchemy(app)

# # --- DATABASE MODEL ---
# class User(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     username = db.Column(db.String(80), unique=True, nullable=False)
#     email = db.Column(db.String(120), unique=True, nullable=False)
#     password_hash = db.Column(db.String(256), nullable=False)

# # --- Global Variables & Model Loading ---
# ml_model = None
# scaler = None
# feature_info = {}
# image_model = None # This will hold the TensorFlow model
# class_labels = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']

# def load_models():
#     """Loads the Scikit-learn model and the TensorFlow model."""
#     global ml_model, scaler, feature_info, image_model
    
#     # Load the traditional Scikit-learn ML model
#     try:
#         print("Loading clinical data model (Scikit-learn)...")
#         ml_model = joblib.load('Alzheimer_detection.pkl')
#         scaler = joblib.load('scaler.pkl')
#         with open('feature_names.json', 'r') as f:
#             feature_info = json.load(f)
#         print("Clinical data model and scaler loaded successfully.")
#     except Exception as e:
#         print(f"Error loading clinical data model: {e}")

#     # Load the TensorFlow image model
#     try:
#         print("Loading image classification model (TensorFlow)...")
#         model_path = "improved_alzheimers_model.keras"
#         image_model = tf.keras.models.load_model(model_path)
#         print("TensorFlow image model loaded successfully.")
#     except Exception as e:
#         print(f"Error loading TensorFlow image model: {e}")
#         print("WARNING: Image prediction will not be available.")

# # --- Helper Functions ---
# def login_required(f):
#     """Decorator to ensure a user is logged in."""
#     def decorated_function(*args, **kwargs):
#         if 'user_id' not in session:
#             flash('Please log in to access this page.', 'error')
#             return redirect(url_for('login'))
#         return f(*args, **kwargs)
#     decorated_function.__name__ = f.__name__
#     return decorated_function

# def preprocess_image(image_path):
#     """Preprocesses an image for the TensorFlow/Keras model."""
#     try:
#         img = load_img(image_path, target_size=(256, 256))
#         img_array = img_to_array(img)
#         img_array = np.expand_dims(img_array, axis=0) / 255.0
#         return img_array
#     except Exception as e:
#         print(f"Error preprocessing image: {e}")
#         return None

# def predict_from_image(image_path):
#     """Makes a prediction using the loaded TensorFlow model."""
#     if image_model is None:
#         return "Image model not loaded", 0.0

#     img_array = preprocess_image(image_path)
#     if img_array is None:
#         return "Image processing error", 0.0
    
#     predictions = image_model.predict(img_array)
#     score = np.max(predictions[0])
#     predicted_idx = np.argmax(predictions[0])
    
#     predicted_label = class_labels[predicted_idx]
#     confidence_score = score * 100
#     return predicted_label, confidence_score

# # --- Authentication & Core Routes ---
# @app.route('/register', methods=['GET', 'POST'])
# def register():
#     if request.method == 'POST':
#         username = request.form.get('username')
#         email = request.form.get('email')
#         password = request.form.get('password')
#         if User.query.filter_by(username=username).first() or User.query.filter_by(email=email).first():
#             flash('Username or email already exists.', 'error')
#             return redirect(url_for('register'))
        
#         hashed_password = generate_password_hash(password)
#         new_user = User(username=username, email=email, password_hash=hashed_password)
#         db.session.add(new_user)
#         db.session.commit()
#         flash('Registration successful! Please log in.', 'success')
#         return redirect(url_for('login'))
#     return render_template('register.html')

# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         user = User.query.filter_by(username=request.form['username']).first()
#         if user and check_password_hash(user.password_hash, request.form['password']):
#             session['user_id'] = user.id
#             session['username'] = user.username
#             flash('Login successful!', 'success')
#             return redirect(url_for('dashboard'))
#         flash('Invalid username or password.', 'error')
#     return render_template('login.html')

# @app.route('/logout')
# def logout():
#     session.clear()
#     flash('You have been logged out.', 'success')
#     return redirect(url_for('home'))

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/dashboard')
# @login_required
# def dashboard():
#     return render_template('dashboard.html')

# @app.route('/form')
# @login_required
# def form():
#     return render_template('form.html', feature_columns=feature_info.get('original_features', []))

# @app.route('/upload')
# @login_required
# def upload():
#     return render_template('upload.html')

# # --- Prediction Routes ---
# @app.route('/predict_ml', methods=['POST'])
# @login_required
# def predict_ml():
#     if ml_model is None or scaler is None:
#         return render_template('error.html', error="Clinical model not loaded.")
#     try:
#         features = {col: request.form.get(col) for col in feature_info.get('original_features', [])}
#         input_df = pd.DataFrame([features])
        
#         input_df[feature_info['numerical_columns']] = input_df[feature_info['numerical_columns']].apply(pd.to_numeric, errors='coerce').fillna(0)
#         input_df_encoded = pd.get_dummies(input_df, columns=feature_info.get('categorical_columns', []))
#         input_df_reindexed = input_df_encoded.reindex(columns=feature_info['encoded_features'], fill_value=0)
        
#         input_array = scaler.transform(input_df_reindexed)
#         prediction_idx = ml_model.predict(input_array)[0]
        
#         diagnosis = "Demented" if prediction_idx == 1 else "Non-Demented"
        
#         return render_template('result.html', diagnosis=diagnosis, features=features)
#     except Exception as e:
#         traceback.print_exc()
#         return render_template('error.html', error=str(e))

# @app.route('/predict_image', methods=['POST'])
# @login_required
# def predict_image():
#     if 'file' not in request.files:
#         return render_template('error.html', error="No file part in request.")
    
#     file = request.files['file']
#     if file.filename == '':
#         return render_template('error.html', error="No file selected.")

#     if file:
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)

#         diagnosis, confidence = predict_from_image(filepath)

#         return render_template('image_result.html', 
#                                diagnosis=diagnosis, 
#                                confidence=f"{confidence:.2f}", 
#                                image_filename=filename)
#     return redirect(url_for('upload'))

# # --- CHATBOT ROUTE ---
# @app.route('/chat', methods=['POST'])
# @login_required
# def chat():
#     user_message = request.json.get('message')
#     if not user_message: return jsonify({"error": "No message provided"}), 400

#     if not OPENROUTER_API_KEY or "YOUR_OPENROUTER_API_KEY_HERE" in OPENROUTER_API_KEY:
#         return jsonify({"reply": "The AI assistant is not configured by the server owner."})

#     system_prompt = "You are NeuroBot, a friendly and supportive AI assistant for the NeuroDetect web app. Help users understand Alzheimer's disease and how to use the app. Answer questions clearly and simply. Do not provide medical advice. If asked for a diagnosis, advise the user to use the app's tools and consult a healthcare professional."
    
#     try:
#         response = requests.post(
#             url="https://openrouter.ai/api/v1/chat/completions",
#             headers={ "Authorization": f"Bearer {OPENROUTER_API_KEY}" },
#             json={
#                 "model": "google/gemini-flash-1.5",
#                 "messages": [
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": user_message}
#                 ]
#             }
#         )
#         response.raise_for_status()
#         reply = response.json()['choices'][0]['message']['content']
#         return jsonify({"reply": reply})
#     except requests.exceptions.HTTPError as e:
#         print(f"Chatbot HTTP Error: {e.response.status_code} - {e.response.text}")
#         return jsonify({"reply": "There was an authentication error with the AI service. Please check the API key."})
#     except Exception as e:
#         print(f"Chatbot Error: {e}")
#         return jsonify({"reply": "Sorry, I'm unable to respond right now."})

# # --- UTILITY PROCESSOR ---
# @app.context_processor
# def utility_processor():
#     """Makes the now() function available in all templates."""
#     def now():
#         return datetime.now()
#     return dict(now=now)

# if __name__ == '__main__':
#     with app.app_context():
#         db.create_all()
#     load_models()
#     app.run(debug=True)
# from flask import Flask, request, render_template, jsonify, redirect, url_for, flash, session
# import pandas as pd
# import numpy as np
# import joblib
# import json
# import os
# import sys
# import traceback
# import tensorflow as tf
# import requests
# from werkzeug.utils import secure_filename
# from werkzeug.security import generate_password_hash, check_password_hash
# from tensorflow.keras.utils import load_img, img_to_array
# from flask_sqlalchemy import SQLAlchemy
# from datetime import datetime

# # --- CONFIGURATION ---
# OPENROUTER_API_KEY = "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"
# FLASK_SECRET_KEY = 'a-very-strong-and-random-secret-key-that-you-should-change'

# # --- APP SETUP ---
# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'static/uploads'
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# app.secret_key = FLASK_SECRET_KEY
# db = SQLAlchemy(app)

# # --- DATABASE MODEL ---
# class User(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     username = db.Column(db.String(80), unique=True, nullable=False)
#     email = db.Column(db.String(120), unique=True, nullable=False)
#     password_hash = db.Column(db.String(256), nullable=False)

# # --- Global Variables & Model Loading ---
# ml_model = None
# scaler = None
# feature_info = {}
# image_model = None
# class_labels = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']

# def load_models():
#     global ml_model, scaler, feature_info, image_model
#     try:
#         print("Loading clinical data model (Scikit-learn)...")
#         ml_model = joblib.load('Alzheimer_detection.pkl')
#         scaler = joblib.load('scaler.pkl')
#         with open('feature_names.json', 'r') as f:
#             feature_info = json.load(f)
#         print("Clinical data model and scaler loaded successfully.")
#     except Exception as e:
#         print(f"Error loading clinical data model: {e}")
#     try:
#         print("Loading image classification model (TensorFlow)...")
#         image_model = tf.keras.models.load_model("improved_alzheimers_model.keras")
#         print("TensorFlow image model loaded successfully.")
#     except Exception as e:
#         print(f"Error loading TensorFlow image model: {e}")

# # --- Helper Functions ---
# def login_required(f):
#     def decorated_function(*args, **kwargs):
#         if 'user_id' not in session:
#             flash('Please log in to access this page.', 'error')
#             return redirect(url_for('login'))
#         return f(*args, **kwargs)
#     decorated_function.__name__ = f.__name__
#     return decorated_function

# def preprocess_image(image_path):
#     try:
#         img = load_img(image_path, target_size=(256, 256))
#         img_array = img_to_array(img)
#         img_array = np.expand_dims(img_array, axis=0) / 255.0
#         return img_array
#     except Exception as e:
#         print(f"Error preprocessing image: {e}")
#         return None

# def predict_from_image(image_path):
#     if image_model is None: return "Image model not loaded", 0.0
#     img_array = preprocess_image(image_path)
#     if img_array is None: return "Image processing error", 0.0
    
#     predictions = image_model.predict(img_array)
#     score = np.max(predictions[0])
#     predicted_idx = np.argmax(predictions[0])
    
#     predicted_label = class_labels[predicted_idx]
#     # EDIT: Return the raw float, not a formatted string
#     confidence_score = score * 100
#     return predicted_label, confidence_score

# # --- Authentication & Core Routes ---
# @app.route('/register', methods=['GET', 'POST'])
# def register():
#     # ... (code remains the same)
#     if request.method == 'POST':
#         username = request.form.get('username')
#         email = request.form.get('email')
#         password = request.form.get('password')
#         if User.query.filter_by(username=username).first() or User.query.filter_by(email=email).first():
#             flash('Username or email already exists.', 'error')
#             return redirect(url_for('register'))
        
#         hashed_password = generate_password_hash(password)
#         new_user = User(username=username, email=email, password_hash=hashed_password)
#         db.session.add(new_user)
#         db.session.commit()
#         flash('Registration successful! Please log in.', 'success')
#         return redirect(url_for('login'))
#     return render_template('register.html')


# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     # ... (code remains the same)
#     if request.method == 'POST':
#         user = User.query.filter_by(username=request.form['username']).first()
#         if user and check_password_hash(user.password_hash, request.form['password']):
#             session['user_id'] = user.id
#             session['username'] = user.username
#             flash('Login successful!', 'success')
#             return redirect(url_for('dashboard'))
#         flash('Invalid username or password.', 'error')
#     return render_template('login.html')

# @app.route('/logout')
# def logout():
#     session.clear()
#     flash('You have been logged out.', 'success')
#     return redirect(url_for('home'))

# @app.route('/')
# def home(): return render_template('index.html')

# @app.route('/dashboard')
# @login_required
# def dashboard(): return render_template('dashboard.html')

# @app.route('/form')
# @login_required
# def form(): return render_template('form.html', feature_columns=feature_info.get('original_features', []))

# @app.route('/upload')
# @login_required
# def upload(): return render_template('upload.html')

# # --- Prediction Routes ---
# @app.route('/predict_ml', methods=['POST'])
# @login_required
# def predict_ml():
#     if ml_model is None or scaler is None:
#         return render_template('error.html', error="Clinical model not loaded.")
#     try:
#         features = {col: request.form.get(col) for col in feature_info.get('original_features', [])}
#         input_df = pd.DataFrame([features])
        
#         input_df[feature_info['numerical_columns']] = input_df[feature_info['numerical_columns']].apply(pd.to_numeric, errors='coerce').fillna(0)
#         input_df_encoded = pd.get_dummies(input_df, columns=feature_info.get('categorical_columns', []))
#         input_df_reindexed = input_df_encoded.reindex(columns=feature_info['encoded_features'], fill_value=0)
        
#         input_array = scaler.transform(input_df_reindexed)
#         prediction_idx = ml_model.predict(input_array)[0]
#         diagnosis = "Demented" if prediction_idx == 1 else "Non-Demented"
        
#         return render_template('result.html', diagnosis=diagnosis, features=features)
#     except Exception as e:
#         traceback.print_exc()
#         return render_template('error.html', error=str(e))

# @app.route('/predict_image', methods=['POST'])
# @login_required
# def predict_image():
#     if 'file' not in request.files:
#         return render_template('error.html', error="No file part in request.")
    
#     file = request.files['file']
#     if file.filename == '':
#         return render_template('error.html', error="No file selected.")

#     if file:
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)

#         diagnosis, confidence = predict_from_image(filepath)

#         return render_template('image_result.html', 
#                                diagnosis=diagnosis, 
#                                confidence=confidence, # Pass the raw float
#                                image_filename=filename)
#     return redirect(url_for('upload'))

# # ... (rest of the file, including chat route and main execution, remains the same)
# @app.route('/chat', methods=['POST'])
# @login_required
# def chat():
#     user_message = request.json.get('message')
#     if not user_message: return jsonify({"error": "No message provided"}), 400

#     if not OPENROUTER_API_KEY or "YOUR_OPENROUTER_API_KEY_HERE" in OPENROUTER_API_KEY:
#         return jsonify({"reply": "The AI assistant is not configured by the server owner."})

#     system_prompt = "You are NeuroBot, a friendly and supportive AI assistant for the NeuroDetect web app. Help users understand Alzheimer's disease and how to use the app. Answer questions clearly and simply. Do not provide medical advice. If asked for a diagnosis, advise the user to use the app's tools and consult a healthcare professional."
    
#     try:
#         response = requests.post(
#             url="https://openrouter.ai/api/v1/chat/completions",
#             headers={ "Authorization": f"Bearer {OPENROUTER_API_KEY}" },
#             json={
#                 "model": "google/gemini-flash-1.5",
#                 "messages": [
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": user_message}
#                 ]
#             }
#         )
#         response.raise_for_status()
#         reply = response.json()['choices'][0]['message']['content']
#         return jsonify({"reply": reply})
#     except requests.exceptions.HTTPError as e:
#         print(f"Chatbot HTTP Error: {e.response.status_code} - {e.response.text}")
#         return jsonify({"reply": "There was an authentication error with the AI service. Please check the API key."})
#     except Exception as e:
#         print(f"Chatbot Error: {e}")
#         return jsonify({"reply": "Sorry, I'm unable to respond right now."})

# @app.context_processor
# def utility_processor():
#     def now():
#         return datetime.now()
#     return dict(now=now)

# if __name__ == '__main__':
#     with app.app_context():
#         db.create_all()
#     load_models()
#     app.run(debug=True)
# from flask import Flask, request, render_template, jsonify, redirect, url_for, flash, session
# import pandas as pd
# import numpy as np
# import joblib
# import json
# import os
# import sys
# import traceback
# import tensorflow as tf
# import requests
# from werkzeug.utils import secure_filename
# from werkzeug.security import generate_password_hash, check_password_hash
# from tensorflow.keras.utils import load_img, img_to_array
# from flask_sqlalchemy import SQLAlchemy
# from datetime import datetime
# from dotenv import load_dotenv # Import the dotenv library

# # Load environment variables from .env file
# load_dotenv()

# # --- CONFIGURATION ---
# # EDIT: Load the API key from the environment
# OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY", 'a-default-strong-secret-key') # Also good practice for the secret key

# # --- APP SETUP ---
# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'static/uploads'
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# app.secret_key = FLASK_SECRET_KEY
# db = SQLAlchemy(app)

# # --- DATABASE MODEL ---
# class User(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     username = db.Column(db.String(80), unique=True, nullable=False)
#     email = db.Column(db.String(120), unique=True, nullable=False)
#     password_hash = db.Column(db.String(256), nullable=False)

# # --- Global Variables & Model Loading ---
# ml_model = None
# scaler = None
# feature_info = {}
# image_model = None
# class_labels = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']

# def load_models():
#     global ml_model, scaler, feature_info, image_model
#     try:
#         print("Loading clinical data model (Scikit-learn)...")
#         ml_model = joblib.load('Alzheimer_detection.pkl')
#         scaler = joblib.load('scaler.pkl')
#         with open('feature_names.json', 'r') as f:
#             feature_info = json.load(f)
#         print("Clinical data model and scaler loaded successfully.")
#     except Exception as e:
#         print(f"Error loading clinical data model: {e}")
#     try:
#         print("Loading image classification model (TensorFlow)...")
#         image_model = tf.keras.models.load_model("improved_alzheimers_model.keras")
#         print("TensorFlow image model loaded successfully.")
#     except Exception as e:
#         print(f"Error loading TensorFlow image model: {e}")

# # --- Helper Functions ---
# def login_required(f):
#     def decorated_function(*args, **kwargs):
#         if 'user_id' not in session:
#             flash('Please log in to access this page.', 'error')
#             return redirect(url_for('login'))
#         return f(*args, **kwargs)
#     decorated_function.__name__ = f.__name__
#     return decorated_function

# def preprocess_image(image_path):
#     try:
#         img = load_img(image_path, target_size=(256, 256))
#         img_array = img_to_array(img)
#         img_array = np.expand_dims(img_array, axis=0) / 255.0
#         return img_array
#     except Exception as e:
#         print(f"Error preprocessing image: {e}")
#         return None

# def predict_from_image(image_path):
#     if image_model is None: return "Image model not loaded", 0.0
#     img_array = preprocess_image(image_path)
#     if img_array is None: return "Image processing error", 0.0
    
#     predictions = image_model.predict(img_array)
#     score = np.max(predictions[0])
#     predicted_idx = np.argmax(predictions[0])
    
#     predicted_label = class_labels[predicted_idx]
#     confidence_score = score * 100
#     return predicted_label, confidence_score

# # --- Authentication & Core Routes ---
# @app.route('/register', methods=['GET', 'POST'])
# def register():
#     # ... (code remains the same)
#     if request.method == 'POST':
#         username = request.form.get('username')
#         email = request.form.get('email')
#         password = request.form.get('password')
#         if User.query.filter_by(username=username).first() or User.query.filter_by(email=email).first():
#             flash('Username or email already exists.', 'error')
#             return redirect(url_for('register'))
        
#         hashed_password = generate_password_hash(password)
#         new_user = User(username=username, email=email, password_hash=hashed_password)
#         db.session.add(new_user)
#         db.session.commit()
#         flash('Registration successful! Please log in.', 'success')
#         return redirect(url_for('login'))
#     return render_template('register.html')


# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     # ... (code remains the same)
#     if request.method == 'POST':
#         user = User.query.filter_by(username=request.form['username']).first()
#         if user and check_password_hash(user.password_hash, request.form['password']):
#             session['user_id'] = user.id
#             session['username'] = user.username
#             flash('Login successful!', 'success')
#             return redirect(url_for('dashboard'))
#         flash('Invalid username or password.', 'error')
#     return render_template('login.html')

# @app.route('/logout')
# def logout():
#     session.clear()
#     flash('You have been logged out.', 'success')
#     return redirect(url_for('home'))

# @app.route('/')
# def home(): return render_template('index.html')

# @app.route('/dashboard')
# @login_required
# def dashboard(): return render_template('dashboard.html')

# @app.route('/form')
# @login_required
# def form(): return render_template('form.html', feature_columns=feature_info.get('original_features', []))

# @app.route('/upload')
# @login_required
# def upload(): return render_template('upload.html')

# # --- Prediction Routes ---
# @app.route('/predict_ml', methods=['POST'])
# @login_required
# def predict_ml():
#     if ml_model is None or scaler is None:
#         return render_template('error.html', error="Clinical model not loaded.")
#     try:
#         features = {col: request.form.get(col) for col in feature_info.get('original_features', [])}
#         input_df = pd.DataFrame([features])
        
#         input_df[feature_info['numerical_columns']] = input_df[feature_info['numerical_columns']].apply(pd.to_numeric, errors='coerce').fillna(0)
#         input_df_encoded = pd.get_dummies(input_df, columns=feature_info.get('categorical_columns', []))
#         input_df_reindexed = input_df_encoded.reindex(columns=feature_info['encoded_features'], fill_value=0)
        
#         input_array = scaler.transform(input_df_reindexed)
#         prediction_idx = ml_model.predict(input_array)[0]
#         diagnosis = "Demented" if prediction_idx == 1 else "Non-Demented"
        
#         return render_template('result.html', diagnosis=diagnosis, features=features)
#     except Exception as e:
#         traceback.print_exc()
#         return render_template('error.html', error=str(e))

# @app.route('/predict_image', methods=['POST'])
# @login_required
# def predict_image():
#     if 'file' not in request.files:
#         return render_template('error.html', error="No file part in request.")
    
#     file = request.files['file']
#     if file.filename == '':
#         return render_template('error.html', error="No file selected.")

#     if file:
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)

#         diagnosis, confidence = predict_from_image(filepath)

#         return render_template('image_result.html', 
#                                diagnosis=diagnosis, 
#                                confidence=confidence,
#                                image_filename=filename)
#     return redirect(url_for('upload'))

# # --- CHATBOT ROUTE ---
# @app.route('/chat', methods=['POST'])
# @login_required
# def chat():
#     user_message = request.json.get('message')
#     if not user_message: return jsonify({"error": "No message provided"}), 400

#     if not OPENROUTER_API_KEY:
#         return jsonify({"reply": "The AI assistant is not configured. The API key is missing."})

#     system_prompt = "You are NeuroBot, a friendly and supportive AI assistant for the NeuroDetect web app. Help users understand Alzheimer's disease and how to use the app. Answer questions clearly and simply. Do not provide medical advice. If asked for a diagnosis, advise the user to use the app's tools and consult a healthcare professional."
    
#     try:
#         response = requests.post(
#             url="https://openrouter.ai/api/v1/chat/completions",
#             headers={ "Authorization": f"Bearer {OPENROUTER_API_KEY}" },
#             json={
#                 "model": "google/gemini-flash-1.5",
#                 "messages": [
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": user_message}
#                 ]
#             }
#         )
#         response.raise_for_status()
#         reply = response.json()['choices'][0]['message']['content']
#         return jsonify({"reply": reply})
#     except requests.exceptions.HTTPError as e:
#         print(f"Chatbot HTTP Error: {e.response.status_code} - {e.response.text}")
#         return jsonify({"reply": "There was an authentication error with the AI service. Please check the API key."})
#     except Exception as e:
#         print(f"Chatbot Error: {e}")
#         return jsonify({"reply": "Sorry, I'm unable to respond right now."})

# @app.context_processor
# def utility_processor():
#     def now():
#         return datetime.now()
#     return dict(now=now)

# if __name__ == '__main__':
#     with app.app_context():
#         db.create_all()
#     load_models()
#     app.run(debug=True)
# from flask import Flask, request, render_template, jsonify, redirect, url_for, flash, session
# import pandas as pd
# import numpy as np
# import joblib
# import json
# import os
# import sys
# import traceback
# import tensorflow as tf
# import requests
# from werkzeug.utils import secure_filename
# from werkzeug.security import generate_password_hash, check_password_hash
# from tensorflow.keras.utils import load_img, img_to_array
# from flask_sqlalchemy import SQLAlchemy
# from datetime import datetime
# from dotenv import load_dotenv

# # Load environment variables from .env file
# load_dotenv()

# # --- CONFIGURATION ---
# OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY", 'a-default-strong-secret-key')

# # --- APP SETUP ---
# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'static/uploads'
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# app.secret_key = FLASK_SECRET_KEY
# db = SQLAlchemy(app)

# # --- DATABASE MODEL ---
# class User(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     username = db.Column(db.String(80), unique=True, nullable=False)
#     email = db.Column(db.String(120), unique=True, nullable=False)
#     password_hash = db.Column(db.String(256), nullable=False)

# # --- Global Variables & Model Loading ---
# ml_model = None
# scaler = None
# feature_info = {}
# image_model = None
# class_labels = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']

# def load_models():
#     global ml_model, scaler, feature_info, image_model
#     try:
#         print("Loading clinical data model (Scikit-learn)...")
#         ml_model = joblib.load('Alzheimer_detection.pkl')
#         scaler = joblib.load('scaler.pkl')
#         with open('feature_names.json', 'r') as f:
#             feature_info = json.load(f)
#         print("Clinical data model and scaler loaded successfully.")
#     except Exception as e:
#         print(f"Error loading clinical data model: {e}")
#     try:
#         print("Loading image classification model (TensorFlow)...")
#         image_model = tf.keras.models.load_model("improved_alzheimers_model.keras")
#         print("TensorFlow image model loaded successfully.")
#     except Exception as e:
#         print(f"Error loading TensorFlow image model: {e}")

# # --- Helper Functions ---
# def login_required(f):
#     def decorated_function(*args, **kwargs):
#         if 'user_id' not in session:
#             flash('Please log in to access this page.', 'error')
#             return redirect(url_for('login'))
#         return f(*args, **kwargs)
#     decorated_function.__name__ = f.__name__
#     return decorated_function

# def preprocess_image(image_path):
#     try:
#         img = load_img(image_path, target_size=(256, 256))
#         img_array = img_to_array(img)
#         img_array = np.expand_dims(img_array, axis=0) / 255.0
#         return img_array
#     except Exception as e:
#         print(f"Error preprocessing image: {e}")
#         return None

# def predict_from_image(image_path):
#     if image_model is None: return "Image model not loaded", 0.0
#     img_array = preprocess_image(image_path)
#     if img_array is None: return "Image processing error", 0.0
    
#     predictions = image_model.predict(img_array)
#     score = np.max(predictions[0])
#     predicted_idx = np.argmax(predictions[0])
    
#     predicted_label = class_labels[predicted_idx]
#     confidence_score = score * 100
#     return predicted_label, confidence_score

# # --- Authentication & Core Routes ---
# @app.route('/register', methods=['GET', 'POST'])
# def register():
#     if request.method == 'POST':
#         username = request.form.get('username')
#         email = request.form.get('email')
#         password = request.form.get('password')
#         if User.query.filter_by(username=username).first() or User.query.filter_by(email=email).first():
#             flash('Username or email already exists.', 'error')
#             return redirect(url_for('register'))
        
#         hashed_password = generate_password_hash(password)
#         new_user = User(username=username, email=email, password_hash=hashed_password)
#         db.session.add(new_user)
#         db.session.commit()
#         flash('Registration successful! Please log in.', 'success')
#         return redirect(url_for('login'))
#     return render_template('register.html')


# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         user = User.query.filter_by(username=request.form['username']).first()
#         if user and check_password_hash(user.password_hash, request.form['password']):
#             session['user_id'] = user.id
#             session['username'] = user.username
#             flash('Login successful!', 'success')
#             return redirect(url_for('dashboard'))
#         flash('Invalid username or password.', 'error')
#     return render_template('login.html')

# @app.route('/logout')
# def logout():
#     session.clear()
#     flash('You have been logged out.', 'success')
#     return redirect(url_for('home'))

# @app.route('/')
# def home(): return render_template('index.html')

# @app.route('/dashboard')
# @login_required
# def dashboard(): return render_template('dashboard.html')

# @app.route('/form')
# @login_required
# def form(): return render_template('form.html', feature_columns=feature_info.get('original_features', []))

# @app.route('/upload')
# @login_required
# def upload(): return render_template('upload.html')

# # --- Prediction Routes ---
# @app.route('/predict_ml', methods=['POST'])
# @login_required
# def predict_ml():
#     if ml_model is None or scaler is None:
#         return render_template('error.html', error="Clinical model not loaded.")
#     try:
#         features = {col: request.form.get(col) for col in feature_info.get('original_features', [])}
#         input_df = pd.DataFrame([features])
        
#         input_df[feature_info['numerical_columns']] = input_df[feature_info['numerical_columns']].apply(pd.to_numeric, errors='coerce').fillna(0)
#         input_df_encoded = pd.get_dummies(input_df, columns=feature_info.get('categorical_columns', []))
#         input_df_reindexed = input_df_encoded.reindex(columns=feature_info['encoded_features'], fill_value=0)
        
#         input_array = scaler.transform(input_df_reindexed)
#         prediction_idx = ml_model.predict(input_array)[0]
#         diagnosis = "Demented" if prediction_idx == 1 else "Non-Demented"
        
#         return render_template('result.html', diagnosis=diagnosis, features=features)
#     except Exception as e:
#         traceback.print_exc()
#         return render_template('error.html', error=str(e))

# @app.route('/predict_image', methods=['POST'])
# @login_required
# def predict_image():
#     if 'file' not in request.files:
#         return render_template('error.html', error="No file part in request.")
    
#     file = request.files['file']
#     if file.filename == '':
#         return render_template('error.html', error="No file selected.")

#     if file:
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)

#         diagnosis, confidence = predict_from_image(filepath)

#         return render_template('image_result.html', 
#                                diagnosis=diagnosis, 
#                                confidence=confidence,
#                                image_filename=filename)
#     return redirect(url_for('upload'))

# # --- CHATBOT ROUTE ---
# @app.route('/chat', methods=['POST'])
# @login_required
# def chat():
#     user_message = request.json.get('message')
#     if not user_message: return jsonify({"error": "No message provided"}), 400

#     if not OPENROUTER_API_KEY:
#         return jsonify({"reply": "The AI assistant is not configured. The API key is missing."})

#     # EDIT: Updated the system prompt to be more restrictive
#     system_prompt = (
#         "You are NeuroBot, a specialized AI assistant for the NeuroDetect Alzheimer's detection application. "
#         "Your ONLY function is to answer questions directly related to Alzheimer's disease, its symptoms, stages, and risk factors. "
#         "You must refuse to answer any questions outside of this topic, including but not limited to general knowledge, personal opinions, or other medical conditions. "
#         "If a user asks a question unrelated to Alzheimer's, you must politely decline and state that your purpose is strictly limited to providing information about Alzheimer's disease. "
#         "Under no circumstances should you provide medical advice or a diagnosis."
#     )
    
#     try:
#         response = requests.post(
#             url="https://openrouter.ai/api/v1/chat/completions",
#             headers={ "Authorization": f"Bearer {OPENROUTER_API_KEY}" },
#             json={
#                 "model": "google/gemini-flash-1.5",
#                 "messages": [
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": user_message}
#                 ]
#             }
#         )
#         response.raise_for_status()
#         reply = response.json()['choices'][0]['message']['content']
#         return jsonify({"reply": reply})
#     except requests.exceptions.HTTPError as e:
#         print(f"Chatbot HTTP Error: {e.response.status_code} - {e.response.text}")
#         return jsonify({"reply": "There was an authentication error with the AI service. Please check the API key."})
#     except Exception as e:
#         print(f"Chatbot Error: {e}")
#         return jsonify({"reply": "Sorry, I'm unable to respond right now."})

# @app.context_processor
# def utility_processor():
#     def now():
#         return datetime.now()
#     return dict(now=now)

# if __name__ == '__main__':
#     with app.app_context():
#         db.create_all()
#     load_models()
#     app.run(debug=True)
from flask import Flask, request, render_template, jsonify, redirect, url_for, flash, session
import pandas as pd
import numpy as np
import joblib
import json
import os
import sys
import traceback
import tensorflow as tf
import requests
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from tensorflow.keras.utils import load_img, img_to_array
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- CONFIGURATION ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY", 'a-default-strong-secret-key')

# --- APP SETUP ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = FLASK_SECRET_KEY
db = SQLAlchemy(app)

# --- DATABASE MODEL ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)

# --- Global Variables & Model Loading ---
ml_model = None
scaler = None
feature_info = {}
image_model = None
class_labels = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']

def load_models():
    global ml_model, scaler, feature_info, image_model
    try:
        print("Loading clinical data model (Scikit-learn)...")
        ml_model = joblib.load('Alzheimer_detection.pkl')
        scaler = joblib.load('scaler.pkl')
        with open('feature_names.json', 'r') as f:
            feature_info = json.load(f)
        print("Clinical data model and scaler loaded successfully.")
    except Exception as e:
        print(f"Error loading clinical data model: {e}")
    try:
        print("Loading image classification model (TensorFlow)...")
        image_model = tf.keras.models.load_model("improved_alzheimers_model.keras")
        print("TensorFlow image model loaded successfully.")
    except Exception as e:
        print(f"Error loading TensorFlow image model: {e}")

# --- Helper Functions ---
def login_required(f):
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

def preprocess_image(image_path):
    try:
        img = load_img(image_path, target_size=(256, 256))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_from_image(image_path):
    if image_model is None: return "Image model not loaded", 0.0
    img_array = preprocess_image(image_path)
    if img_array is None: return "Image processing error", 0.0
    
    predictions = image_model.predict(img_array)
    score = np.max(predictions[0])
    predicted_idx = np.argmax(predictions[0])
    
    predicted_label = class_labels[predicted_idx]
    confidence_score = score * 100
    return predicted_label, confidence_score

# --- Authentication & Core Routes ---
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        if User.query.filter_by(username=username).first() or User.query.filter_by(email=email).first():
            flash('Username or email already exists.', 'error')
            return redirect(url_for('register'))
        
        hashed_password = generate_password_hash(password)
        new_user = User(username=username, email=email, password_hash=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username']).first()
        if user and check_password_hash(user.password_hash, request.form['password']):
            session['user_id'] = user.id
            session['username'] = user.username
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        flash('Invalid username or password.', 'error')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'success')
    return redirect(url_for('home'))

@app.route('/')
def home(): return render_template('index.html')

@app.route('/dashboard')
@login_required
def dashboard(): return render_template('dashboard.html')

# EDIT: Add the new route for the about page
@app.route('/about')
@login_required
def about():
    return render_template('about.html')

@app.route('/form')
@login_required
def form(): return render_template('form.html', feature_columns=feature_info.get('original_features', []))

@app.route('/upload')
@login_required
def upload(): return render_template('upload.html')

# --- Prediction Routes ---
@app.route('/predict_ml', methods=['POST'])
@login_required
def predict_ml():
    if ml_model is None or scaler is None:
        return render_template('error.html', error="Clinical model not loaded.")
    try:
        features = {col: request.form.get(col) for col in feature_info.get('original_features', [])}
        input_df = pd.DataFrame([features])
        
        input_df[feature_info['numerical_columns']] = input_df[feature_info['numerical_columns']].apply(pd.to_numeric, errors='coerce').fillna(0)
        input_df_encoded = pd.get_dummies(input_df, columns=feature_info.get('categorical_columns', []))
        input_df_reindexed = input_df_encoded.reindex(columns=feature_info['encoded_features'], fill_value=0)
        
        input_array = scaler.transform(input_df_reindexed)
        prediction_idx = ml_model.predict(input_array)[0]
        diagnosis = "Demented" if prediction_idx == 1 else "Non-Demented"
        
        return render_template('result.html', diagnosis=diagnosis, features=features)
    except Exception as e:
        traceback.print_exc()
        return render_template('error.html', error=str(e))

@app.route('/predict_image', methods=['POST'])
@login_required
def predict_image():
    if 'file' not in request.files:
        return render_template('error.html', error="No file part in request.")
    
    file = request.files['file']
    if file.filename == '':
        return render_template('error.html', error="No file selected.")

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        diagnosis, confidence = predict_from_image(filepath)

        return render_template('image_result.html', 
                               diagnosis=diagnosis, 
                               confidence=confidence,
                               image_filename=filename)
    return redirect(url_for('upload'))

# --- CHATBOT ROUTE ---
@app.route('/chat', methods=['POST'])
@login_required
def chat():
    user_message = request.json.get('message')
    if not user_message: return jsonify({"error": "No message provided"}), 400

    if not OPENROUTER_API_KEY:
        return jsonify({"reply": "The AI assistant is not configured. The API key is missing."})

    system_prompt = (
        "You are NeuroBot, a specialized AI assistant for the NeuroDetect Alzheimer's detection application. "
        "Your ONLY function is to answer questions directly related to Alzheimer's disease, its symptoms, stages, and risk factors. "
        "You must refuse to answer any questions outside of this topic. "
        "When providing information, especially lists (e.g., symptoms, stages), you MUST format your response using Markdown. Use asterisks for bullet points. "
        "If a user asks a question unrelated to Alzheimer's, you must politely decline and state that your purpose is strictly limited to providing information about Alzheimer's disease. "
        "Under no circumstances should you provide medical advice or a diagnosis."
    )
    
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={ "Authorization": f"Bearer {OPENROUTER_API_KEY}" },
            json={
                "model": "google/gemini-flash-1.5",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ]
            }
        )
        response.raise_for_status()
        reply = response.json()['choices'][0]['message']['content']
        return jsonify({"reply": reply})
    except requests.exceptions.HTTPError as e:
        print(f"Chatbot HTTP Error: {e.response.status_code} - {e.response.text}")
        return jsonify({"reply": "There was an authentication error with the AI service. Please check the API key."})
    except Exception as e:
        print(f"Chatbot Error: {e}")
        return jsonify({"reply": "Sorry, I'm unable to respond right now."})

@app.context_processor
def utility_processor():
    def now():
        return datetime.now()
    return dict(now=now)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    load_models()
    app.run(debug=True)

