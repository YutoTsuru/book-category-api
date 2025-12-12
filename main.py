import json
import re
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

ARTIFACTS_DIR = "artifacts"

with open(f"{ARTIFACTS_DIR}/vocab.json", encoding="utf-8") as f:
    vocab = json.load(f)

with open(f"{ARTIFACTS_DIR}/tfidf_params.json", encoding="utf-8") as f:
    tfidf_params = json.load(f)

with open(f"{ARTIFACTS_DIR}/label_map.json", encoding="utf-8") as f:
    label_map = json.load(f)

centers = np.load(f"{ARTIFACTS_DIR}/centers.npy")
center_norms = np.load(f"{ARTIFACTS_DIR}/center_norms.npy")
idf = np.load(f"{ARTIFACTS_DIR}/idf.npy")

token_pattern = re.compile(tfidf_params["token_pattern"])

def tokenize(text):
    return token_pattern.findall(text.lower())

def vectorize(text):
    vec = np.zeros(len(vocab))
    tokens = tokenize(text)
    for t in tokens:
        if t in vocab:
            vec[vocab[t]] += 1
    vec = vec * idf
    norm = np.linalg.norm(vec)
    return vec / norm if norm != 0 else vec

def predict_cluster(vec):
    sims = centers @ vec / center_norms
    return int(np.argmax(sims))

app = FastAPI()

class Item(BaseModel):
    text: str

@app.post("/predict")
def predict(item: Item):
    vec = vectorize(item.text)
    cluster = predict_cluster(vec)
    label = label_map[str(cluster)]
    return {
        "cluster": cluster,
        "label": label
    }

@app.get("/health")
def health():
    return {"status": "ok"}
