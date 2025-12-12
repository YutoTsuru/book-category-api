from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

model = joblib.load("genre_pipeline.joblib")

class Item(BaseModel):
    text: str

@app.post("/predict")
def predict(item: Item):
    pred = model.predict([item.text])
    return {"label": str(pred[0])}
