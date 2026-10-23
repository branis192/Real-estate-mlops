import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(X):
    """Builds the global ColumnTransformer pipeline for the Ames Housing dataset."""
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

    preprocessor = ColumnTransformer(
        transformers=[
            ("num_zero", pipeline_num_zero, num_impute_zero),
            ("num_median", pipeline_num_median, num_impute_median),
            ("cat_none", pipeline_cat_none, cat_impute_none),
            ("cat_freq", pipeline_cat_freq, cat_impute_most_frequent),
        ]
    )

    return preprocessor