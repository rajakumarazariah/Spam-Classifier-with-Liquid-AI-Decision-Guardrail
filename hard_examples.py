import os
import pandas as pd
import joblib
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

pipeline = joblib.load(os.path.join("model", "tfidf_lr_pipeline.joblib"))

df = pd.read_csv("spam.csv", encoding="latin-1", usecols=["v1", "v2"])
df["label"] = df["v1"].str.strip().str.lower().map({"ham": 0, "spam": 1})
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)

_, X_test, _, y_test = train_test_split(
    df["v2"].astype(str), df["label"], test_size=0.20, random_state=42, stratify=df["label"]
)

test_df = pd.DataFrame({"message": X_test, "actual": y_test})
test_df["predicted_prob"] = pipeline.predict_proba(test_df["message"])[:, 1]
test_df["pred"] = (test_df["predicted_prob"] >= 0.5).astype(int)

# Extract 2 False Positives and 2 False Negatives
fps = test_df[(test_df["actual"] == 0) & (test_df["pred"] == 1)].head(2)
fns = test_df[(test_df["actual"] == 1) & (test_df["pred"] == 0)].head(2)
hard_samples = pd.concat([fps, fns])

def explain_failure(message: str, actual: int, pred_prob: float):
    prompt = f"""
    You are an ML debugging engineer. A TF-IDF + Logistic Regression spam classifier failed on this test example.
    
    Message: "{message}"
    True Ground Truth: {"SPAM" if actual == 1 else "HAM (Genuine)"}
    Model Predicted Spam Probability: {pred_prob:.4f}
    
    In 2-3 concise bullet points, explain:
    1. Why did the linear bag-of-words / n-gram representation fail (e.g., misleading keywords, short conversational context)?
    2. What specific architectural or data preprocessing improvement would fix this?
    """

    # Primary model from your available Groq models
    selected_model = "openai/gpt-oss-120b"

    try:
        completion = client.chat.completions.create(
            model=selected_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_completion_tokens=300
        )
        return completion.choices[0].message.content
    except Exception as e:
        # Fallback to 20B or Qwen if rate-limited or busy
        try:
            completion = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_completion_tokens=300
            )
            return completion.choices[0].message.content
        except Exception as fallback_err:
            return f"Groq query failed: {str(fallback_err)}"

print("Analyzing Hard Examples via Groq...\n")
for idx, row in hard_samples.iterrows():
    print(f"--- Message: \"{row['message']}\" ---")
    print(f"Actual: {'SPAM' if row['actual'] == 1 else 'HAM'} | Model Confidence P(Spam): {row['predicted_prob']:.4f}")
    explanation = explain_failure(row["message"], row["actual"], row["predicted_prob"])
    print(explanation)
    print("-" * 60 + "\n")