from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib

# Load model + vectorizer (joblib format)
model = joblib.load("app/model.joblib")
vectorizer = joblib.load("app/vectorizer.joblib")

app = FastAPI()

# Mount static files (CSS, JS)
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "prediction": None})

@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request, caption: str = Form(...)):
    X = vectorizer.transform([caption])
    prediction = model.predict(X)[0]
    label = "🚨 Scam" if prediction == 1 else "✅ Legit"
    return templates.TemplateResponse(
        "index.html", 
        {"request": request, "prediction": label, "caption": caption}
    )
