from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import json
from .utils import prepare_features, get_lr_explanation, preprocess_text

# === Initialize FastAPI ===
app = FastAPI(title="Twitter Scam Detector")

# Mount static files (CSS, JS, images)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Jinja2 templates
templates = Jinja2Templates(directory="app/templates")

# === Load model & vectorizer ===
model = joblib.load("app/logreg_calibrated_model.joblib")
vectorizer = joblib.load("app/tfidf_vectorizer.joblib")

# Try to load per-model thresholds from training, fallback to default
DEFAULT_THRESHOLD = 0.7            # conservative default for labeling as scam
UNCERTAIN_LOWER = 0.45             # below this => likely legit
UNCERTAIN_UPPER = DEFAULT_THRESHOLD # above this => labeled scam; between => uncertain

try:
    with open("app/thresholds.json","r") as f:
        thresholds = json.load(f)  # expected: {"logreg": {"best_threshold": 0.6, ...}, ...}
except Exception:
    thresholds = {}

def get_threshold_for_model(model_name="logreg"):
    entry = thresholds.get(model_name)
    if entry and "best_threshold" in entry:
        return float(entry["best_threshold"])
    return DEFAULT_THRESHOLD

# === API schema ===
class InputText(BaseModel):
    text: str

# helper to make a labeled/uncertain decision and explanation
def predict_and_explain(text):
    text = text.strip()
    X_new = prepare_features([text], vectorizer)
    prob = float(model.predict_proba(X_new)[0][1]) if hasattr(model, "predict_proba") else float(model.decision_function(X_new)[0])
    thr = get_threshold_for_model("logreg")  # change key if using other saved names
    # decide
    if prob >= thr:
        label = 1
        label_text = "Scam"
    elif prob < UNCERTAIN_LOWER:
        label = 0
        label_text = "Legit"
    else:
        # uncertain region
        label = None
        label_text = "Uncertain"

    # explanation for logistic-like models
    explanation = []
    try:
        # attempt to use logistic-style explanation helper
        explanation = get_lr_explanation(text, vectorizer, model, top_k=5)
    except Exception:
        explanation = []

    return {"input": text, "probability": prob, "label": label, "label_text": label_text, "explanation": explanation, "threshold_used": thr}

# === API endpoints ===
@app.get("/api")
def home():
    return {"message": "Welcome to the Twitter Scam Detector API! Use POST /predict to analyze captions."}

@app.post("/api/predict")
def predict_api(data: InputText):
    text = data.text
    if not text or not text.strip():
        return {"error": "Empty text provided."}
    res = predict_and_explain(text)
    return {
        "input": res["input"],
        "prediction": int(res["label"]) if res["label"] is not None else None,
        "scam_probability": float(res["probability"]),
        "result": res["label_text"],
        "explanation": res["explanation"],
        "threshold_used": res["threshold_used"]
    }

# === Frontend routes ===
@app.get("/", response_class=HTMLResponse)
def load_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/", response_class=HTMLResponse)
def predict_form(request: Request, caption: str = Form(...)):
    res = predict_and_explain(caption)
    prob = res["probability"]
    label_text = res["label_text"]
    if label_text == "Uncertain":
        display_result = {
            "input": caption,
            "prediction": None,
            "scam_probability": f"{prob:.2%}",
            "result": "🔎 Uncertain — please review manually",
            "explanation": res["explanation"]
        }
    else:
        display_result = {
            "input": caption,
            "prediction": int(res["label"]) if res["label"] is not None else None,
            "scam_probability": f"{prob:.2%}",
            "result": "⚠️ Scam" if res["label"] == 1 else "✅ Legit",
            "explanation": res["explanation"]
        }
    return templates.TemplateResponse("index.html", {"request": request, "result": display_result})
