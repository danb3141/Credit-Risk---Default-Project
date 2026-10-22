# simple preprocessing and pytorch nn script
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def fit_neural_network(X_train, y_train, epochs=10, batch_size=1024):
    num_cols = X_train.select_dtypes(include="number").columns.tolist()
    cat_cols = X_train.select_dtypes(exclude="number").columns.tolist()

    # Apply scaling and encoding to the respecitve feature types
    scaler = StandardScaler()
    X_numerical = pd.DataFrame(scaler.fit_transform(X_train[num_cols]), columns=num_cols, index=X_train.index)

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_categorical = pd.DataFrame(encoder.fit_transform(X_train[cat_cols]), columns=encoder.get_feature_names_out(cat_cols), index=X_train.index)
    
    # Combine the new features
    X_train = pd.concat([X_numerical, X_categorical], axis=1).values.astype(np.float32)
    y_train = np.asarray(y_train, dtype=np.float32).reshape(-1, 1)

    # validation split to keep an eye on model accuracy
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=10)
    
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train_sub), torch.tensor(y_train_sub)), batch_size=batch_size, shuffle=True)

    # simple 3-layer architecture
    torch.manual_seed(10)
    input_dim = X_train.shape[1]
    model = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.30),
                          nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.20),
                          nn.Linear(64, 1), nn.Sigmoid())
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()

    # Training
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validation + logging
        model.eval()
        with torch.no_grad():
            val_auc = roc_auc_score(y_val, model(torch.tensor(X_val)).numpy())
        
        print(f"epoch: {epoch+1} - train loss: {epoch_loss / len(train_loader):.4f} - val auc: {val_auc:.4f}")

    # Save preprocessing objects directly onto the model to be used during inference, preventing leakage and ensuring correct preprocessing.
    model.scaler_ = scaler
    model.encoder_ = encoder
    model.numeric_cols_ = num_cols
    model.categorical_cols_ = cat_cols
    model.feature_names_ = pd.concat([X_numerical, X_categorical], axis=1).columns.tolist()
    return model

# data transformation for test set inference
def transform_neural_network_data(model, X):
    X_numerical = pd.DataFrame(model.scaler_.transform(X[model.numeric_cols_]), columns=model.numeric_cols_, index=X.index)
    X_categorical = pd.DataFrame(model.encoder_.transform(X[model.categorical_cols_]), columns=model.encoder_.get_feature_names_out(model.categorical_cols_), index=X.index)
    return pd.concat([X_numerical, X_categorical], axis=1).values.astype(np.float32)

def predict_neural_network(model, X):
    X_processed = transform_neural_network_data(model, X)
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X_processed)).numpy().flatten()
