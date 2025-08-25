from flask import Flask, request, render_template, session, redirect, url_for, flash
from flask_mysqldb import MySQL # type: ignore
import bcrypt
import pickle
import joblib
import os
import numpy as np
import logging

# Initialize logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# BMI Encoder Initialization
def initialize_bmi_encoder():
    try:
        bmi_encoder = pickle.load(open('models/bmi_le.pkl', 'rb'))
    except:
        from sklearn.preprocessing import LabelEncoder
        bmi_encoder = LabelEncoder()
        bmi_encoder.fit(['Underweight', 'Normal', 'Overweight', 'Obese'])
        os.makedirs('models', exist_ok=True)
        pickle.dump(bmi_encoder, open('models/bmi_le.pkl', 'wb'))
    return bmi_encoder

# Load Models and Encoders
def load_models():
    models = {}
    encoders = {}
    
    try:
        models['rf'] = joblib.load('models/random_forest_model.pkl')
        models['xgb'] = joblib.load('models/xgboost_model.pkl')
        models['hybrid'] = joblib.load('models/hybrid_model.pkl')
        
        encoders['gender'] = pickle.load(open('models/gen_le.pkl', 'rb'))
        encoders['occupation'] = pickle.load(open('models/occ_le.pkl', 'rb'))
        encoders['bmi'] = initialize_bmi_encoder()
        encoders['sleep'] = pickle.load(open('models/sleep_le.pkl', 'rb'))
        
        return models, encoders
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise

app = Flask(__name__, template_folder='templates')
app.secret_key = "2004@varshini"

# Database Configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'flask'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

mysql = MySQL(app)

# Load models at startup
try:
    models, encoders = load_models()
except Exception as e:
    logger.critical(f"Failed to load models: {str(e)}")
    raise

# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        try:
            email = request.form['email'].strip().lower()
            password = request.form['password'].encode('utf-8')
            
            cur = mysql.connection.cursor()
            cur.execute("SELECT * FROM user WHERE email = %s", (email,))
            user = cur.fetchone()
            cur.close()
            
            if user and bcrypt.checkpw(password, user['password'].encode('utf-8')):
                session["user"] = {
                    "username": user["name"],
                    "email": user["email"]
                }
                return redirect(url_for('predict'))
            
            flash("Invalid credentials!", 'error')
            return redirect(url_for('login'))
            
        except Exception as e:
            logger.error(f"Login error: {str(e)}")
            flash("Login failed. Please try again.", 'error')
            return redirect(url_for('login'))
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            username = request.form['name'].strip()
            email = request.form['email'].strip().lower()
            password = request.form['password'].encode('utf-8')
            
            hashed_password = bcrypt.hashpw(password, bcrypt.gensalt())
            
            cur = mysql.connection.cursor()
            cur.execute("SELECT email FROM user WHERE email = %s", (email,))
            if cur.fetchone():
                cur.close()
                flash("Email already exists!", 'error')
                return redirect(url_for('register'))
            
            cur.execute(
                "INSERT INTO user (name, email, password) VALUES (%s, %s, %s)",
                (username, email, hashed_password)
            )
            mysql.connection.commit()
            cur.close()
            
            flash("Registration successful! Please login.", 'success')
            return redirect(url_for('login'))
            
        except Exception as e:
            mysql.connection.rollback()
            logger.error(f"Registration error: {str(e)}")
            flash("Registration failed. Please try again.", 'error')
            return redirect(url_for('register'))
    
    return render_template('register.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        try:
            # Validate all required fields are present
            required_fields = ['age', 'gender', 'occupation', 'sleep_duration', 
                             'quality_of_sleep', 'physical_activity', 'stress_level',
                             'bmi_category', 'systolic_bp', 'diastolic_bp', 
                             'heart_rate', 'daily_steps']
            
            if not all(field in request.form for field in required_fields):
                flash("Please fill all required fields", 'error')
                return redirect(url_for('predict'))
            
            # Process form data
            data = {
                'age': int(request.form['age']),
                'gender': request.form['gender'],
                'occupation': request.form['occupation'],
                'sleep_duration': float(request.form['sleep_duration']),
                'quality_of_sleep': int(request.form['quality_of_sleep']),
                'physical_activity': int(request.form['physical_activity']),
                'stress_level': int(request.form['stress_level']),
                'bmi_category': request.form['bmi_category'],
                'systolic_bp': int(request.form['systolic_bp']),
                'diastolic_bp': int(request.form['diastolic_bp']),
                'heart_rate': int(request.form['heart_rate']),
                'daily_steps': int(request.form['daily_steps'])
            }
            
            # Validate BMI category
            valid_bmi = ['Underweight', 'Normal', 'Overweight', 'Obese']
            if data['bmi_category'] not in valid_bmi:
                data['bmi_category'] = 'Normal'
            
            # Encode categorical features
            try:
                gender_encoded = encoders['gender'].transform([data['gender']])[0]
                occupation_encoded = encoders['occupation'].transform([data['occupation']])[0]
                
                if data['bmi_category'] not in encoders['bmi'].classes_:
                    encoders['bmi'].classes_ = np.append(encoders['bmi'].classes_, data['bmi_category'])
                bmi_encoded = encoders['bmi'].transform([data['bmi_category']])[0]
                
            except ValueError as e:
                logger.error(f"Encoding error: {str(e)}")
                flash("Invalid input values. Please check your data.", 'error')
                return redirect(url_for('predict'))
            
            # Prepare feature array in correct order
            features = [
                data['age'], gender_encoded, occupation_encoded, data['sleep_duration'],
                data['quality_of_sleep'], data['physical_activity'], data['stress_level'],
                bmi_encoded, data['systolic_bp'], data['diastolic_bp'], 
                data['heart_rate'], data['daily_steps']
            ]
            
            # Make predictions
            predictions = {}
            for model_name, model in models.items():
                try:
                    prediction = model.predict([features])[0]
                    predictions[model_name] = int(prediction) if hasattr(prediction, 'item') else prediction
                except Exception as e:
                    logger.error(f"Prediction error with {model_name}: {str(e)}")
                    predictions[model_name] = "Error"
            
            # Correct prediction mapping (matches backend encoding)
            prediction_map = {
                0: "No Sleep Disorder",
                1: "Sleep Apnea",
                2: "Insomnia"
            }
            
            # Store results
            session['predictions'] = {
                'final': prediction_map.get(predictions['hybrid'], "Unknown"),
                'all': {k: prediction_map.get(v, "Unknown") for k, v in predictions.items()}
            }
            
            return redirect(url_for('prediction_result'))
            
        except Exception as e:
            logger.error(f"Prediction processing error: {str(e)}")
            flash("Prediction failed. Please try again.", 'error')
            return redirect(url_for('predict'))
    
    return render_template('predict.html')

@app.route('/prediction_result')
def prediction_result():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    if 'predictions' not in session:
        flash('No prediction data found. Please submit the form again.', 'error')
        return redirect(url_for('predict'))
    
    return render_template(
        'prediction_result.html',
        prediction=session['predictions']['final'],
        model_results=session['predictions']['all']
    )

@app.route('/logout')
def logout():
    session.pop("user", None)
    flash("You have been logged out successfully.", 'success')
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)