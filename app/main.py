import os
from typing import Any, Dict
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# 1. En haut du fichier avec les imports :
from prometheus_fastapi_instrumentator import Instrumentator


app = FastAPI(
    title="Ames Real Estate Price Prediction API",
    description="MLOps API serving a Linear Regression model to predict house prices.",
    version="1.0.0",
)
# 2. Juste après la ligne app = FastAPI(...) :
Instrumentator().instrument(app).expose(app)

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "models", "model_pipeline.joblib"
)

if os.path.exists(MODEL_PATH):
    model_pipeline = joblib.load(MODEL_PATH)
else:
    raise FileNotFoundError(f"Pipeline artifact not found at {MODEL_PATH}.")


class HouseFeatures(BaseModel):
    features: Dict[str, Any]

    class Config:
        json_schema_extra = {
            "example": {
                "features": {
                    "Gr Liv Area": 1500,
                    "Overall Qual": 7,
                    "Garage Area": 400,
                    "Neighborhood": "NAmes",
                    "Full Bath": 2,
                }
            }
        }


@app.get("/")
def home():
    return {"status": "healthy", "model_loaded": model_pipeline is not None}


@app.post("/predict")
def predict_price(payload: HouseFeatures):
    try:
        input_df = pd.DataFrame([payload.features])

        # Récupère la liste de toutes les colonnes vues à l'entraînement
        expected_cols = model_pipeline.named_steps["preprocessor"].feature_names_in_

        # Ajoute les colonnes manquantes remplies de NaN (gérées par les Imputers)
        for col in expected_cols:
            if col not in input_df.columns:
                input_df[col] = np.nan

        # Réaligne les colonnes dans l'ordre attendu
        input_df = input_df[expected_cols]

        prediction_log = model_pipeline.predict(input_df)
        prediction_dollars = np.expm1(prediction_log)[0]

        return {"predicted_price_usd": round(float(prediction_dollars), 2)}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")
