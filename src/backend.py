import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn import preprocessing
from sklearn.utils import resample  # <-- Added this import
import pickle
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
import os 
import joblib

# Create models folder if not exists
os.makedirs("models", exist_ok=True)

# Load and preprocess data
def load_and_preprocess_data():
    try:
        df = pd.read_csv(r'C:\Users\chitt\Music\ITML06 Sleep disorder\Dataset\Sleep_health_and_lifestyle_dataset.csv')
        
        # Handle missing values
        df['Sleep Disorder'].fillna('None', inplace=True)
        df.drop('Person ID', axis=1, inplace=True)
        
        # Split blood pressure
        df['systolic_bp'] = pd.to_numeric(df['Blood Pressure'].apply(lambda x: x.split('/')[0]))
        df['diastolic_bp'] = pd.to_numeric(df['Blood Pressure'].apply(lambda x: x.split('/')[1]))
        df.drop('Blood Pressure', axis=1, inplace=True)
        
        # Normalize BMI categories
        df['BMI Category'] = df['BMI Category'].replace('Normal Weight', 'Normal')
        
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

# Balance dataset
def balance_dataset(df):
    try:
        none_df = df[df['Sleep Disorder'] == 'None']
        sleep_apnea_df = df[df['Sleep Disorder'] == 'Sleep Apnea']
        insomnia_df = df[df['Sleep Disorder'] == 'Insomnia']
        
        # Upsample minority classes
        none_upsampled = resample(none_df, replace=True, n_samples=500, random_state=42)
        sleep_apnea_upsampled = resample(sleep_apnea_df, replace=True, n_samples=500, random_state=42)
        insomnia_upsampled = resample(insomnia_df, replace=True, n_samples=500, random_state=42)
        
        return pd.concat([none_upsampled, sleep_apnea_upsampled, insomnia_upsampled]).sample(frac=1, random_state=42)
    except Exception as e:
        print(f"Error balancing dataset: {str(e)}")
        raise

# Train and save models
def train_and_save_models(X_train, y_train):
    try:
        # Initialize models
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        hybrid_model = VotingClassifier(
            estimators=[('rf', rf_model), ('xgb', xgb_model)],
            voting='soft'
        )
        
        # Train models
        print("Training Random Forest model...")
        rf_model.fit(X_train, y_train)
        print("Training XGBoost model...")
        xgb_model.fit(X_train, y_train)
        print("Training Hybrid model...")
        hybrid_model.fit(X_train, y_train)
        
        # Save models
        os.makedirs("models", exist_ok=True)
        joblib.dump(rf_model, 'models/random_forest_model.pkl')
        joblib.dump(xgb_model, 'models/xgboost_model.pkl')
        joblib.dump(hybrid_model, 'models/hybrid_model.pkl')
        
        return rf_model, xgb_model, hybrid_model
    except Exception as e:
        print(f"Error training models: {str(e)}")
        raise

# Main execution
if __name__ == '__main__':
    try:
        print("Loading and preprocessing data...")
        df = load_and_preprocess_data()
        
        print("Balancing dataset...")
        upsampled_df = balance_dataset(df)
        
        print("Encoding categorical variables...")
        # Encode categorical variables
        encoders = {}
        for col in ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']:
            enc = preprocessing.LabelEncoder()
            upsampled_df[col] = enc.fit_transform(upsampled_df[col])
            pickle.dump(enc, open(f'models/{col.lower()[:3]}_le.pkl', 'wb'))
            encoders[col] = enc
        
        # Prepare features and target
        X = upsampled_df.drop('Sleep Disorder', axis=1)
        y = upsampled_df['Sleep Disorder']
        
        print("Splitting data into train/test sets...")
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        print("Training models...")
        # Train and save models
        rf_model, xgb_model, hybrid_model = train_and_save_models(X_train, y_train)
        
        print("\nModel Evaluation:")
        # Evaluate models
        for name, model in [('Random Forest', rf_model), 
                           ('XGBoost', xgb_model), 
                           ('Hybrid', hybrid_model)]:
            preds = model.predict(X_test)
            print(f"\n{name} Classification Report:")
            print(classification_report(y_test, preds))
        
        df.to_csv("preprocessed_sleep_data.csv", index=False)
        print("\nPreprocessing and model training complete!")
        
    except Exception as e:
        print(f"\nError in main execution: {str(e)}")