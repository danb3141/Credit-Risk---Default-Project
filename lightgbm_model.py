"""
Simple Preprocessing and LightGBM model implementation
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from lightgbm import LGBMClassifier

def fit_lightgbm(X_train, y_train, class_weight=None):
    # Split the features by their type
    num_cols = X_train.select_dtypes(include="number").columns.tolist()
    cat_cols = X_train.select_dtypes(exclude="number").columns.tolist()

    # Encode categorical features
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_processed = X_train.copy()
    X_processed[cat_cols] = encoder.fit_transform(X_train[cat_cols])

    # Fit a simple LGBM model
    model = LGBMClassifier(objective="binary", n_estimators=500, learning_rate=0.05, num_leaves=31,
                           min_child_samples=50, colsample_bytree=0.8, reg_lambda=1.0,
                           class_weight=class_weight, random_state=10, n_jobs=-1, verbosity=-1)
    
    model.fit(X_processed, y_train)

    # Save preprocessing objects directly onto the model to be used during inference
    model.encoder_ = encoder
    model.numeric_cols_ = num_cols
    model.categorical_cols_ = cat_cols
    model.feature_names_ = X_processed.columns.tolist()

    return model

# Inference time preprocessing functions, transform the test datasets to be ready for model inference
def transform_lightgbm_data(model, X):
    X_processed = X.copy()
    X_processed[model.categorical_cols_] = model.encoder_.transform(X[model.categorical_cols_])
    return X_processed

def predict_lightgbm(model, X):
    X_processed = transform_lightgbm_data(model, X)
    return model.predict_proba(X_processed)[:, 1]

def feature_importance_table(model):
    results = pd.DataFrame({"feature": model.feature_names_, "importance": model.feature_importances_})
    return results.sort_values("importance", ascending=False)
