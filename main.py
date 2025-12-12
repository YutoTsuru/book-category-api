from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

model = joblib.load("models/genre_pipeline.joblib")
label_encoder = joblib.load("models/label_encoder.joblib")

class Item(BaseModel):
    text: str

@app.post("/predict")
def predict(item: Item):
    pred_id = model.predict([item.text])[0]
    genre = label_encoder.inverse_transform([pred_id])[0]
    return {"genre": genre}

@app.get("/health")
def health():
    return {"status": "ok"}