import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression

# 1. Load data
DATA_PATH = "spam.csv"
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset '{DATA_PATH}' not found in root directory.")

# Handle latin-1 encoding common in the SMS spam dataset
df = pd.read_csv(DATA_PATH, encoding="latin-1", usecols=["v1", "v2"])
df = df.dropna(subset=["v1", "v2"])

# Map labels: 'ham' -> 0, 'spam' -> 1
df["label"] = df["v1"].str.strip().str.lower().map({"ham": 0, "spam": 1})
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)

X = df["v2"].astype(str)
y = df["label"]

# 2. Stratified Train-Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 3. Hybrid Word + Subword Char TF-IDF Pipeline
vectorizer = FeatureUnion([
    ("word_tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=6000, sublinear_tf=True)),
    ("char_tfidf", TfidfVectorizer(ngram_range=(3, 5), analyzer="char_wb", max_features=10000, sublinear_tf=True))
])

pipeline = Pipeline([
    ("features", vectorizer),
    ("classifier", LogisticRegression(C=2.5, class_weight="balanced", max_iter=1000, random_state=42))
])

print("Training TF-IDF + Logistic Regression pipeline...")
pipeline.fit(X_train, y_train)

# 4. Save artifacts
os.makedirs("model", exist_ok=True)
model_path = os.path.join("model", "tfidf_lr_pipeline.joblib")
joblib.dump(pipeline, model_path)
print(f"Model saved successfully to {model_path}")