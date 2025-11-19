# src/predict_category.py

import pandas as pd
import numpy as np
import re
import joblib
import os
import sys
from scipy.sparse import hstack

# 1. PATH DEFINITIONS AND ARTIFACT LOADING
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')

# Load the saved transformers and model
try:
    SCALER = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    LE = joblib.load(os.path.join(MODELS_DIR, 'label_encoder.pkl'))
    TFIDF = joblib.load(os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'))
    MODEL = joblib.load(os.path.join(MODELS_DIR, 'product_classifier.pkl'))
except FileNotFoundError as e:
    print(f"ERROR: Could not load necessary model artifacts. Please run train_model.py first.")
    print(f"Missing file: {e}")
    sys.exit(1)

FEATURE_COLS = ['title_length', 'word_count', 'has_storage_unit', 'has_dimension', 'has_digit', 'is_tech_product']


# 2. FEATURE ENGINEERING FUNCTION
def create_engineered_features(title):
    """Creates numerical features from the title (must match training logic)."""
    df_temp = pd.DataFrame({'Product Title': [title]})
    titles = df_temp['Product Title'].str.lower()
    
    # 1. Structural features
    df_temp['title_length'] = titles.apply(len)
    df_temp['word_count'] = titles.apply(lambda x: len(x.split()))

    # 2. Binary Specification features
    storage_pattern = r'(?:\d+)\s*(?:gb|tb|mb|l|kg|litre)\b'
    df_temp['has_storage_unit'] = titles.str.contains(storage_pattern, regex=True).astype(int)
    
    dimension_pattern = r'(?:\d+(?:\.\d+)?)\s*(?:cm|inch|")\b'
    df_temp['has_dimension'] = titles.str.contains(dimension_pattern, regex=True).astype(int)
    
    df_temp['has_digit'] = titles.str.contains(r'\d+', regex=True).astype(int)

    tech_pattern = r'\b(?:ssd|led|usb|hdmi|wifi|ghz|mp|core)\b'
    df_temp['is_tech_product'] = titles.str.contains(tech_pattern, regex=True).astype(int)

    return df_temp[FEATURE_COLS].values


# 3. MAIN PREDICTION FUNCTION
def predict_category(product_title):
    """
    Prepares input and performs prediction using saved transformers and model.
    """
    
    # 1. Feature Engineering
    X_numerical = create_engineered_features(product_title)
    
    # 2. Scaling (Using the saved MinMaxScaler)
    X_numerical_scaled = SCALER.transform(X_numerical)
    
    # 3. Text Vectorization (Using the saved TFIDF)
    X_text = TFIDF.transform([product_title.lower()])
    
    # 4. Combine Matrices
    X_final = hstack([X_text, X_numerical_scaled])
    
    # 5. Prediction
    prediction_encoded = MODEL.predict(X_final)
    
    # 6. Decode Prediction
    predicted_category = LE.inverse_transform(prediction_encoded)
    
    return predicted_category[0]


# 4. COMMAND LINE EXECUTION
if __name__ == '__main__':
    print("--- E-Commerce Product Category Classifier (Prediction Tool) ---")
    
    while True:
        try:
            user_input = input("\nEnter Product Title (or 'quit' to exit): ")
            
            if user_input.lower() in ['quit', 'exit']:
                break
            
            if not user_input.strip():
                print("Please enter a valid product title.")
                continue

            # Perform prediction
            category = predict_category(user_input.strip())
            
            print(f"\n[INPUT]: {user_input.strip()}")
            print(f"-> [PREDICTED CATEGORY]: **{category}**")
            
        except Exception as e:
            print(f"An unexpected error occurred during prediction: {e}")
            break