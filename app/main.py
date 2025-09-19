from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
from .utils import prepare_features

# === Initialize FastAPI ===
app = FastAPI(title="Twitter Scam Detector")

# Mount static files (CSS, JS, images)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Jinja2 templates
templates = Jinja2Templates(directory="app/templates")

# === Load model & vectorizer ===
model = joblib.load("app/svm_calibrated_model.joblib")
vectorizer = joblib.load("app/tfidf_vectorizer.joblib")

# === API schema ===
class InputText(BaseModel):
    text: str

# === API endpoints ===
@app.get("/api")
def home():
    return {"message": "Welcome to the Twitter Scam Detector API! Use POST /predict to analyze captions."}

@app.post("/api/predict")
def predict_api(data: InputText):
    text = data.text.strip()
    if not text:
        return {"error": "Empty text provided."}

    X_new = prepare_features([text], vectorizer)
    pred = model.predict(X_new)[0]
    prob = model.predict_proba(X_new)[0][1]

    return {
        "input": text,
        "prediction": int(pred),
        "scam_probability": float(prob),
        "result": "Scam" if pred == 1 else "Legit"
    }

# === Frontend routes ===
@app.get("/", response_class=HTMLResponse)
def load_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/", response_class=HTMLResponse)
def predict_form(request: Request, caption: str = Form(...)):
    X_new = prepare_features([caption], vectorizer)
    pred = model.predict(X_new)[0]
    prob = model.predict_proba(X_new)[0][1]

    result = {
        "input": caption,
        "prediction": int(pred),
        "scam_probability": f"{prob:.2%}",
        "result": "⚠️ Scam" if pred == 1 else "✅ Legit"
    }
    return templates.TemplateResponse("index.html", {"request": request, "result": result})
