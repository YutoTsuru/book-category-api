from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

vectorizer = joblib.load("models/vectorizer.joblib")
kmeans = joblib.load("models/kmeans.pkl")

class Item(BaseModel):
    text: str

@app.post("/predict")
def predict(item: Item):
    X = vectorizer.transform([item.text])
    cluster = int(kmeans.predict(X)[0])
    return {"prediction": cluster}

@app.get("/health")
def health():
    return {"status": "ok"}
