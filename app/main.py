import os
from typing import Dict, Any
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 1. Initialize FastAPI app
app = FastAPI(
    title="Ames Real Estate Price Prediction API",
    description="MLOps API serving a Linear Regression model to predict house prices.",
    version="1.0.0"
)

# 2. Define the path to our trained pipeline artifact
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "model_pipeline.joblib")

# Load the pipeline once when the server starts
if os.path.exists(MODEL_PATH):
    model_pipeline = joblib.load(MODEL_PATH)
    print("🎯 ML pipeline loaded successfully in FastAPI!")
else:
    raise FileNotFoundError(f"Pipeline artifact not found at {MODEL_PATH}. Please run src/train.py first.")


# 3. Define the input data schema using Pydantic
class HouseFeatures(BaseModel):
    # We accept a dynamic dictionary of features matching the AmesHousing columns
    features: Dict[str, Any]

    class Config:
        json_schema_extra = {
            "example": {
                "features": {
                    "Gr Liv Area": 1500,
                    "Overall Qual": 7,
                    "Garage Area": 400,
                    "Neighborhood": "NAmes",
                    "Full Bath": 2
                    # The pipeline will safely handle any missing columns!
                }
            }
        }


# 4. Create the health check endpoint
@app.get("/")
def home():
    return {"status": "healthy", "model_loaded": model_pipeline is not None}


# 5. Create the prediction endpoint
@app.post("/predict")
def predict_price(payload: HouseFeatures):
    try:
        # Convert the incoming JSON features into a Pandas DataFrame (1 row)
        input_df = pd.DataFrame([payload.features])
        
        # Ensure all columns expected by the model exist (fill missing ones with NaN)
        # Our pipeline's SimpleImputer will automatically handle these NaNs!
        prediction_log = model_pipeline.predict(input_df)
        
        # Inverse the log transformation to get the price in actual USD ($)
        prediction_dollars = np.expm1(prediction_log)[0]
        
        return {
            "predicted_price_usd": round(float(prediction_dollars), 2)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")