"""
train_model.py
---------------
Trains a baseline Fake News detector using TF-IDF + Logistic Regression.

Dataset expected (Kaggle "Fake and Real News Dataset"):
  data/Fake.csv
  data/True.csv
Download from: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
Place both CSVs inside a `data/` folder next to this script.

Usage:
    python train_model.py
Outputs:
    model.joblib       -> trained classifier
    vectorizer.joblib  -> fitted TF-IDF vectorizer
    metrics.json       -> accuracy / precision / recall / f1 on held-out test set
"""

import json
import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
FAKE_CSV = os.path.join(DATA_DIR, "Fake.csv")
TRUE_CSV = os.path.join(DATA_DIR, "True.csv")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), "vectorizer.joblib")
METRICS_PATH = os.path.join(os.path.dirname(__file__), "metrics.json")


def load_data() -> pd.DataFrame:
    if not (os.path.exists(FAKE_CSV) and os.path.exists(TRUE_CSV)):
        raise FileNotFoundError(
            f"Expected Fake.csv and True.csv inside {DATA_DIR}. "
            "Download the Kaggle Fake and Real News dataset and place them there."
        )

    fake_df = pd.read_csv(FAKE_CSV)
    true_df = pd.read_csv(TRUE_CSV)

    fake_df["label"] = 0  # 0 = fake
    true_df["label"] = 1  # 1 = real

    df = pd.concat([fake_df, true_df], ignore_index=True)

    # Combine title + text for a stronger signal
    df["content"] = (df.get("title", "").fillna("") + " " + df.get("text", "").fillna(""))
    df = df[["content", "label"]].dropna()
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    return df


def train():
    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} rows.")

    X_train, X_test, y_train, y_test = train_test_split(
        df["content"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    print("Vectorizing (TF-IDF)...")
    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=2,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training Logistic Regression...")
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train_vec, y_train)

    print("Evaluating...")
    preds = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, target_names=["fake", "real"], output_dict=True)

    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, target_names=["fake", "real"]))

    joblib.dump(clf, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump({"accuracy": acc, "report": report}, f, indent=2)

    print(f"\nSaved model -> {MODEL_PATH}")
    print(f"Saved vectorizer -> {VECTORIZER_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    train()
