import os
import requests
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Spam Classifier with Liquid AI Decision Guardrail", version="1.0.0")

MODEL_PATH = os.path.join("model", "tfidf_lr_pipeline.joblib")
if not os.path.exists(MODEL_PATH):
    raise RuntimeError("Model pipeline not loaded. Train the model using train.py first.")

pipeline = joblib.load(MODEL_PATH)
LIQUID_API_KEY = os.getenv("LIQUID_API_KEY")

class TextInput(BaseModel):
    message: str = Field(..., min_length=1, example="Urgent! Your account requires verification. Click https://secure-bank.xyz")

class PredictionOutput(BaseModel):
    label: str
    is_spam: bool
    confidence: float
    decision_engine: str
    guardrail_note: str

def query_liquid_decision(text: str) -> float:
    """
    Calls Liquid AI's structured d1.1 System One decision endpoint.
    Returns calibrated probability of spam via the 'noul' primitive.
    """
    if not LIQUID_API_KEY:
        raise ValueError("LIQUID_API_KEY not configured in environment.")

    url = "https://api.liquid.ai/decisions/v1/systemone"
    headers = {
        "Authorization": f"Bearer {LIQUID_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "d1.1",
        "state": text,
        "questions": {
            "is_spam": {
                "type": "noul",
                "instructions": "Is this text message an unsolicited promotional advertisement, phishing attempt, lottery scam, or spam?"
            }
        }
    }
    
    response = requests.post(url, headers=headers, json=payload, timeout=3.0)
    
    # If the API returned an HTTP error, capture the message
    if response.status_code != 200:
        raise RuntimeError(f"Liquid API error {response.status_code}: {response.text}")

    data = response.json()

    # Liquid AI d1 schema returns the evaluation in 'answers'
    if "answers" in data and "is_spam" in data["answers"]:
        answer_obj = data["answers"]["is_spam"]
        # noul returns the probability directly under 'noul' or 'probability'
        if "noul" in answer_obj:
            return float(answer_obj["noul"])
        if "probability" in answer_obj:
            return float(answer_obj["probability"])

    # Fallback key checks for alternate proxy implementations
    if "decisions" in data and "is_spam" in data["decisions"]:
        dec_obj = data["decisions"]["is_spam"]
        return float(dec_obj.get("probability", dec_obj.get("noul", 0.5)))

    raise KeyError(f"Unexpected response schema from Liquid API: {data}")

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: TextInput):
    text = input_data.message.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Input message cannot be empty.")

    # 1. Local TF-IDF + Logistic Regression inference
    prob_spam = float(pipeline.predict_proba([text])[0][1])

    # 2. Confidence band check: High certainty vs. Ambiguous
    # Ambiguity interval: between 0.35 and 0.70
    if 0.35 <= prob_spam <= 0.70:
        try:
            liquid_prob = query_liquid_decision(text)
            final_is_spam = liquid_prob >= 0.50
            return PredictionOutput(
                label="spam" if final_is_spam else "ham",
                is_spam=final_is_spam,
                confidence=round(liquid_prob, 4),
                decision_engine="Liquid AI d1.1 (System One Decision Model)",
                guardrail_note=f"Triggered due to low local confidence (P(spam)={prob_spam:.3f}). Re-evaluated via structured zero-token decision."
            )
        except Exception as e:
            # Fallback to local prediction if decision API fails or key is missing
            final_is_spam = prob_spam >= 0.50
            return PredictionOutput(
                label="spam" if final_is_spam else "ham",
                is_spam=final_is_spam,
                confidence=round(prob_spam, 4),
                decision_engine="Local TF-IDF + LR (Guardrail Fallback)",
                guardrail_note=f"Guardrail attempt failed: {str(e)}"
            )

    # 3. High confidence: resolved locally
    final_is_spam = prob_spam >= 0.50
    confidence = prob_spam if final_is_spam else (1.0 - prob_spam)
    return PredictionOutput(
        label="spam" if final_is_spam else "ham",
        is_spam=final_is_spam,
        confidence=round(confidence, 4),
        decision_engine="Local TF-IDF + Logistic Regression",
        guardrail_note="High certainty prediction; bypassed external decision model."
    )

@app.get("/health")
def health():
    return {"status": "ok", "model_ready": True}