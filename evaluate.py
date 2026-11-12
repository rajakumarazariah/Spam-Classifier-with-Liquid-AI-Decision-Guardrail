import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

MODEL_PATH = os.path.join("model", "tfidf_lr_pipeline.joblib")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model not found. Run train.py first.")

df = pd.read_csv("spam.csv", encoding="latin-1", usecols=["v1", "v2"])
df["label"] = df["v1"].str.strip().str.lower().map({"ham": 0, "spam": 1})
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)

X = df["v2"].astype(str)
y = df["label"]

_, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

pipeline = joblib.load(MODEL_PATH)

# Inference
y_probs = pipeline.predict_proba(X_test)[:, 1]
y_pred = (y_probs >= 0.5).astype(int)

# Metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_probs)

print("=" * 45)
print("           EVALUATION RESULTS")
print("=" * 45)
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-Score : {f1:.4f}")
print(f"ROC-AUC  : {auc:.4f}")
print("\nConfusion Matrix [TN, FP / FN, TP]:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))