'''
Simple Preprocessing and LR model implementation
'''

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

def fit_logistic(X_train, y_train, class_weight=None, C=1.0, solver='lbfgs'):
    # Split the features by their type
    num_cols = X_train.select_dtypes(include="number").columns
    cat_cols = X_train.select_dtypes(exclude="number").columns

    # Apply scaling and encoding to the respecitve feature types
    num_scaler = StandardScaler()
    X_numerical = pd.DataFrame(num_scaler.fit_transform(X_train[num_cols]), columns=num_cols, index=X_train.index)

    cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_categorical = pd.DataFrame(cat_encoder.fit_transform(X_train[cat_cols]), columns=cat_encoder.get_feature_names_out(cat_cols), index=X_train.index)

    # Combine the new features
    X_processed = pd.concat([X_numerical, X_categorical], axis=1)
    model = LogisticRegression(max_iter=300, C=C, solver=solver, class_weight=class_weight)
    model.fit(X_processed, y_train)

    # Save preprocessing objects directly onto the model to be used during inference, preventing leakage and ensuring correct preprocessing.
    model.scaler_ = num_scaler
    model.encoder_ = cat_encoder
    model.numeric_cols_ = num_cols.tolist()
    model.categorical_cols_ = cat_cols.tolist()
    model.feature_names_ = X_processed.columns.tolist()

    return model

# Inference time preprocessing functions, transform the test datasets to be ready for model inference.
def transform_logistic_data(model, X):
    X_numerical = pd.DataFrame(model.scaler_.transform(X[model.numeric_cols_]), columns=model.numeric_cols_, index=X.index)
    X_categorical = pd.DataFrame(model.encoder_.transform(X[model.categorical_cols_]), columns=model.encoder_.get_feature_names_out(model.categorical_cols_), index=X.index)
    return pd.concat([X_numerical, X_categorical], axis=1)

def predict_logistic(model, X):
    X_processed = transform_logistic_data(model, X)
    return model.predict_proba(X_processed)[:, 1]

def coefficient_table(model):
    results = pd.DataFrame({"feature": model.feature_names_,
                            "coefficient": model.coef_[0],
                            "odds_ratio": np.exp(model.coef_[0]),
                            "abs_coeff": np.abs(model.coef_[0])})
    return results.sort_values("abs_coeff", ascending=False)
