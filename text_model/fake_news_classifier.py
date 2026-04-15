import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import re
from difflib import SequenceMatcher

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_ROOT / "models" / "tanzania_fact_model.pkl"
VECTORIZER_PATH = REPO_ROOT / "models" / "tanzania_vectorizer.pkl"
DATA_PATH = REPO_ROOT / "data" / "tanzania_publicinfo_dataset.csv"

def load_training_data():
    """Load training data from CSV"""
    if not DATA_PATH.exists():
        print("❌ Training data not found. Run scraper.py first.")
        return None, None

    df = pd.read_csv(DATA_PATH)

    # Filter out unverified claims for training
    df = df[df['label'].isin(['verified', 'false'])]

    if len(df) < 10:
        print("⚠️ Not enough training data. Need at least 10 labeled examples.")
        return None, None

    return df['statement'].fillna(''), df['label']

def preprocess_text(text):
    """Basic text preprocessing"""
    if not isinstance(text, str):
        return ""


    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def train_and_evaluate():
    print("🤖 Training Tanzania Fact Verification Model...")

    X, y = load_training_data()
    if X is None:
        return None

    
    X_processed = X.apply(preprocess_text)

    
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )

    
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    
    clf = MultinomialNB()
    clf.fit(X_train_vec, y_train)

    
    y_pred = clf.predict(X_test_vec)
    print("\n📊 Model Performance:")
    print(classification_report(y_test, y_pred))

    # Save model and vectorizer
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print(f"💾 Model saved to {MODEL_PATH}")
    print(f"💾 Vectorizer saved to {VECTORIZER_PATH}")

    return clf

def load_model():
    """Load trained model and vectorizer"""
    try:
        clf = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        return clf, vectorizer
    except FileNotFoundError:
        print("❌ Model not found. Run train_and_evaluate() first.")
        return None, None

def predict_statement(text):
    """Predict if a statement is verified or false"""
    clf, vectorizer = load_model()
    if clf is None:
        return {"verified": 0.0, "false": 0.0, "unverified": 1.0}

    # A single-class model (e.g. trained on an all-"verified" dataset) can only
    # ever return its one class at 100%. Refuse to answer rather than hand out
    # a meaningless "verified" stamp.
    if len(getattr(clf, "classes_", [])) < 2:
        return {"verified": 0.0, "false": 0.0, "unverified": 1.0}

    # Preprocess input
    processed_text = preprocess_text(text)

    # Vectorize
    text_vec = vectorizer.transform([processed_text])

    # Get probabilities
    proba = clf.predict_proba(text_vec)[0]

    # Map to our labels
    class_labels = clf.classes_
    result = {}

    for i, label in enumerate(class_labels):
        result[label] = float(proba[i])

    # Add unverified if not present
    if 'unverified' not in result:
        result['unverified'] = 0.0

    return result

def save_model(model):
    """Save model (legacy function for compatibility)"""
    if model:
        joblib.dump(model, MODEL_PATH)
        print(f"💾 Model saved to {MODEL_PATH}")
    else:
        print("❌ No model to save")

if __name__ == "__main__":
    # Train the model
    model = train_and_evaluate()
    if model:
        # Test prediction
        test_text = "Tanzania has over 60 million people"
        scores = predict_statement(test_text)
        print(f"\n🧪 Test prediction for: '{test_text}'")
        print(f"Scores: {scores}")