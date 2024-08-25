from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List

app = FastAPI()

# Load the trained model
model = joblib.load('../models/model.pkl')

# Serve static files
app.mount("/static", StaticFiles(directory="front_end"), name="static")

class PredictRequest(BaseModel):
    features: List[float]

@app.get("/")
def read_root():
    return {"message": "Welcome to the Housing Price Prediction API!"}

@app.post("/predict")
def predict(request: PredictRequest):
    features = request.features
    if len(features) != 3:
        raise HTTPException(status_code=400, detail="Invalid number of features. Expected 3 features.")
    
    try:
        features_array = np.array(features, dtype=float).reshape(1, -1)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid feature values. All values should be numeric.")
    
    prediction = model.predict(features_array)
    return {"prediction": prediction[0]}
