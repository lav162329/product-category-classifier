# src/train_model.py

import pandas as pd
import numpy as np
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler 
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from scipy.sparse import hstack
import os

# 1. PATH DEFINITIONS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'products.csv')
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')
MODEL_FILENAME = 'product_classifier.pkl'

# Define the columns created by Feature Engineering
FEATURE_COLS = ['title_length', 'word_count', 'has_storage_unit', 'has_dimension', 'has_digit', 'is_tech_product']

# 2. DATA PREPARATION FUNCTIONS

def load_and_clean_data(data_path):
    """Loads data and performs essential cleaning."""
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()
    df.dropna(subset=['Product Title', 'Category Label'], inplace=True)
    df['Product Title'] = df['Product Title'].astype(str).str.lower()
    df.reset_index(drop=True, inplace=True)
    return df

def create_engineered_features(df):
    """Creates numerical features (Feature Engineering) using non-capturing groups."""
    titles = df['Product Title']
    
    # Structural and Binary features (using the same compatible RegEx)
    df['title_length'] = titles.apply(len)
    df['word_count'] = titles.apply(lambda x: len(x.split()))
    
    storage_pattern = r'(?:\d+)\s*(?:gb|tb|mb|l|kg|litre)\b'
    df['has_storage_unit'] = titles.str.contains(storage_pattern, regex=True).astype(int)
    
    dimension_pattern = r'(?:\d+(?:\.\d+)?)\s*(?:cm|inch|")\b'
    df['has_dimension'] = titles.str.contains(dimension_pattern, regex=True).astype(int)
    
    df['has_digit'] = titles.str.contains(r'\d+', regex=True).astype(int)

    tech_pattern = r'\b(?:ssd|led|usb|hdmi|wifi|ghz|mp|core)\b'
    df['is_tech_product'] = titles.str.contains(tech_pattern, regex=True).astype(int)

    return df

def transform_and_combine_features(df, feature_cols):
    """Scales, vectorizes, and combines features."""
    
    # Target Encoding
    le = LabelEncoder()
    df['Category_Encoded'] = le.fit_transform(df['Category Label'])
    
    # Scaling numerical features (Using MinMaxScaler for non-negative output)
    scaler = MinMaxScaler() 
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # Text Vectorization (TF-IDF)
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
    X_text = tfidf_vectorizer.fit_transform(df['Product Title'])

    # Combining Features
    X_numerical = df[feature_cols].values
    X_final = hstack([X_text, X_numerical])
    y = df['Category_Encoded']

    # Saving transformation artifacts
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
    joblib.dump(le, os.path.join(MODELS_DIR, 'label_encoder.pkl'))
    joblib.dump(tfidf_vectorizer, os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'))
    
    return X_final, y, le.classes_

# 3. MAIN TRAINING FUNCTION

def train_and_save_model():
    """Main function to execute the entire training pipeline."""
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # 1. Data Preparation
    df = load_and_clean_data(DATA_PATH)
    df = create_engineered_features(df)
    X_final, y, classes = transform_and_combine_features(df, FEATURE_COLS)
    
    # Split: 90% Train, 10% Test (for script performance check)
    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.1, random_state=42, stratify=y)
    
    print(f"\nTraining Model on {X_train.shape[0]} samples (Linear SVC)...")

    # 2. Model Training (Optimal model from the notebook)
    model = LinearSVC(random_state=42, C=0.5, max_iter=5000)
    model.fit(X_train, y_train)
    
    # 3. Evaluation on the small test set
    y_pred = model.predict(X_test)
    print("\nEvaluation on 10% Test Set:")
    print(classification_report(y_test, y_pred, target_names=classes, zero_division=0))
    
    # 4. Saving the model
    model_path = os.path.join(MODELS_DIR, MODEL_FILENAME)
    joblib.dump(model, model_path)
    print(f"\nModel saved successfully to: {model_path}")
    
    print("\n--- Training Pipeline Complete ---")

if __name__ == '__main__':
    train_and_save_model()