import os
import sys

# Assure la résolution des imports locaux
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor


def main():
    print("⏳ Starting ML training pipeline...")

    data_path = "data/AmesHousing.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Run script from root directory."
        )

    df = pd.read_csv(data_path)
    df = df[df["Gr Liv Area"] < 4000].reset_index(drop=True)

    X = df.drop(columns=["SalePrice", "Order", "PID"])
    y = np.log1p(df["SalePrice"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor(X)

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )

    print("🤖 Training Linear Regression baseline model...")
    model_pipeline.fit(X_train, y_train)

    y_pred_log = model_pipeline.predict(X_test)
    y_test_dollars = np.expm1(y_test)
    y_pred_dollars = np.expm1(y_pred_log)

    mae = mean_absolute_error(y_test_dollars, y_pred_dollars)
    print(f"📏 Model Performance (MAE): ${mae:.2f}")

    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_output_path = os.path.join(model_dir, "model_pipeline.joblib")

    joblib.dump(model_pipeline, model_output_path)
    print(f"💾 Production pipeline successfully saved to: {model_output_path}")


if __name__ == "__main__":
    main()