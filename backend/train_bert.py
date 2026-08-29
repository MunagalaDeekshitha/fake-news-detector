"""
train_bert.py
--------------
Fine-tunes DistilBERT on the Fake/Real news dataset.

This is the "upgrade" step from the TF-IDF + Logistic Regression baseline
(train_model.py) to a real transformer model. Expect this to take:
  - ~10-20 minutes on a decent NVIDIA GPU
  - ~1-3+ hours on CPU only (laptop-dependent)

To keep CPU-only training reasonable, this script trains on a SUBSET of the
data by default (SAMPLE_SIZE below). If you have a GPU, bump SAMPLE_SIZE up
(or set it to None to use the full dataset) for a stronger model.

Usage:
    python train_bert.py

Outputs (saved into ./bert_model/):
    config.json, model.safetensors, tokenizer files, etc.
    bert_metrics.json  -> accuracy/precision/recall/f1 on held-out test set
"""

import json
import os

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# ---- Config ----
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
FAKE_CSV = os.path.join(DATA_DIR, "Fake.csv")
TRUE_CSV = os.path.join(DATA_DIR, "True.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "bert_model")
METRICS_PATH = os.path.join(os.path.dirname(__file__), "bert_metrics.json")

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256          # tokens per example — longer = slower but more context
SAMPLE_SIZE = 6000        # set to None to use the full ~44,898 rows (needs a GPU realistically)
EPOCHS = 2
BATCH_SIZE = 8            # lower this (e.g. 4) if you hit "out of memory" errors


def load_data() -> pd.DataFrame:
    if not (os.path.exists(FAKE_CSV) and os.path.exists(TRUE_CSV)):
        raise FileNotFoundError(f"Expected Fake.csv and True.csv inside {DATA_DIR}.")

    fake_df = pd.read_csv(FAKE_CSV)
    true_df = pd.read_csv(TRUE_CSV)
    fake_df["label"] = 0  # fake
    true_df["label"] = 1  # real

    df = pd.concat([fake_df, true_df], ignore_index=True)
    df["content"] = (df.get("title", "").fillna("") + " " + df.get("text", "").fillna(""))
    df = df[["content", "label"]].dropna()
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    if SAMPLE_SIZE is not None:
        # Keep classes balanced when subsampling
        df = (
            df.groupby("label", group_keys=False)
            .apply(lambda x: x.sample(min(len(x), SAMPLE_SIZE // 2), random_state=42))
            .sample(frac=1, random_state=42)
            .reset_index(drop=True)
        )
    return df


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cpu":
        print(
            "No GPU detected — training on CPU. This will be slower. "
            f"Using a subset of {SAMPLE_SIZE} rows to keep training time reasonable."
        )

    print("Loading data...")
    df = load_data()
    print(f"Training on {len(df)} rows.")

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )

    print(f"Loading tokenizer/model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    def tokenize(batch):
        return tokenizer(
            batch["content"], truncation=True, padding="max_length", max_length=MAX_LENGTH
        )

    train_ds = Dataset.from_pandas(train_df[["content", "label"]])
    test_ds = Dataset.from_pandas(test_df[["content", "label"]])
    train_ds = train_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)

    training_args = TrainingArguments(
        output_dir=os.path.join(os.path.dirname(__file__), "bert_checkpoints"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to=[],  # disable wandb/etc prompts
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    print("Training...")
    trainer.train()

    print("Evaluating...")
    preds_output = trainer.predict(test_ds)
    preds = np.argmax(preds_output.predictions, axis=-1)
    labels = preds_output.label_ids

    acc = accuracy_score(labels, preds)
    report = classification_report(labels, preds, target_names=["fake", "real"], output_dict=True)
    print(f"Final accuracy: {acc:.4f}")
    print(classification_report(labels, preds, target_names=["fake", "real"]))

    print(f"Saving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    with open(METRICS_PATH, "w") as f:
        json.dump({"accuracy": acc, "report": report, "sample_size": len(df), "device": device}, f, indent=2)

    print("Done. Update app.py to load from bert_model/ to use this model (see README).")


if __name__ == "__main__":
    main()
