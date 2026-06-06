import os
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(X):
    """Builds the global ColumnTransformer pipeline for the Ames Housing dataset."""

    # 1. Group columns by imputation strategy
    num_impute_zero = [
        "Mas Vnr Area",
        "BsmtFin SF 1",
        "BsmtFin SF 2",
        "Bsmt Unf SF",
        "Total Bsmt SF",
        "Bsmt Full Bath",
        "Bsmt Half Bath",
        "Garage Cars",
        "Garage Area",
    ]

    num_impute_median = [
        c
        for c in X.select_dtypes(include=[np.number]).columns
        if c not in num_impute_zero
    ]

    cat_impute_none = [
        "Alley",
        "Mas Vnr Type",
        "Bsmt Qual",
        "Bsmt Cond",
        "Bsmt Exposure",
        "BsmtFin Type 1",
        "BsmtFin Type 2",
        "Fireplace Qu",
        "Garage Type",
        "Garage Finish",
        "Garage Qual",
        "Garage Cond",
        "Pool QC",
        "Fence",
        "Misc Feature",
    ]

    cat_impute_most_frequent = [
        c
        for c in X.select_dtypes(include=["object", "string"]).columns
        if c not in cat_impute_none
    ]

    # 2. Create sub-pipelines
    pipeline_num_zero = Pipeline(
        steps=[
            ("imputer_zero", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler()),
        ]
    )

    pipeline_num_median = Pipeline(
        steps=[
            ("imputer_median", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    pipeline_cat_none = Pipeline(
        steps=[
            (
                "imputer_none",
                SimpleImputer(strategy="constant", fill_value="None"),
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    pipeline_cat_freq = Pipeline(
        steps=[
            ("imputer_freq", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    # 3. Assemble into global ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num_zero", pipeline_num_zero, num_impute_zero),
            ("num_median", pipeline_num_median, num_impute_median),
            ("cat_none", pipeline_cat_none, cat_impute_none),
            ("cat_freq", pipeline_cat_freq, cat_impute_most_frequent),
        ]
    )

    return preprocessor

def main():
    print("⏳ Starting ML training pipeline...")

    # 1. Load data & filter extreme outliers
    data_path = "data/AmesHousing.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Run script from root directory."
        )

    df = pd.read_csv(data_path)
    df = df[df["Gr Liv Area"] < 4000].reset_index(drop=True)

    # Separate features and target (log scale)
    X = df.drop(columns=["SalePrice", "Order", "PID"])
    y = np.log1p(df["SalePrice"])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 2. Get preprocessor and build global model pipeline
    preprocessor = build_preprocessor(X)

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )

    # 3. Train the model
    print("🤖 Training Linear Regression baseline model...")
    model_pipeline.fit(X_train, y_train)

    # 4. Evaluate the model
    y_pred_log = model_pipeline.predict(X_test)
    y_test_dollars = np.expm1(y_test)
    y_pred_dollars = np.expm1(y_pred_log)

    mae = mean_absolute_error(y_test_dollars, y_pred_dollars)
    print(f"📏 Model Performance (MAE): ${mae:.2f}")

    # 5. Save the production-ready pipeline artifact
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_output_path = os.path.join(model_dir, "model_pipeline.joblib")

    joblib.dump(model_pipeline, model_output_path)
    print(f"💾 Production pipeline successfully saved to: {model_output_path}")


if __name__ == "__main__":
    main()