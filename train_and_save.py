# print("1. Loading and preparing data...")

import pandas as pd
import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pathlib
import joblib
import numpy as np

# --- Configuration (must match app.py) ---
FEATURE_COLS = ["gold_diff", "exp_diff", "kills_diff", "dragons_diff", "deaths_diff"]
CLASS_COL = 'blueWins'
RANDOM_SEED = 33
DL_ENCODING_DIM = 8
DL_PATIENCE = 500
DL_LR = 0.001
DL_NEURONS = 8
DL_ACTIVATION = 'tanh'
DL_OUTPUT_ACTIVATION = 'sigmoid'
DL_LOSS = 'mse'
DL_METRIC = 'Accuracy'
BASE_MAX_EPOCH = 10 * DL_PATIENCE # 5000


# --- Data Loading and Preprocessing ---
print("1. Loading and preparing data...")
games = pd.read_csv("high_diamond_ranked_10min.csv")

# Feature Engineering
best_features = pd.DataFrame()
best_features[CLASS_COL] = games[CLASS_COL]
best_features["gold_diff"] = games["blueGoldDiff"]
best_features["exp_diff"] = games["blueExperienceDiff"]
best_features["kills_diff"] = games["blueKills"] - games["redKills"]
best_features["dragons_diff"] = games["blueDragons"] - games["redDragons"]
best_features["deaths_diff"] = games["blueDeaths"] - games["redDeaths"]

X = best_features[FEATURE_COLS].to_numpy()
Y_base = best_features[CLASS_COL].to_numpy() # For baseline (1D array)
n_classes = 2 # 'blueWins' is binary

# Normalize X and split data
X_scaler = StandardScaler()
X_scaler.fit(X)
X_scaled = X_scaler.transform(X)

X_train_scaled, X_val_scaled, Y_train_base, _ = train_test_split(
    X_scaled, Y_base, test_size=0.2, random_state=RANDOM_SEED
)

# Prepare Y for DL model (One-Hot Encoded) - only used for training
Y_dl = keras.utils.to_categorical(Y_base, num_classes=n_classes)

X_train_dl_scaled, X_val_dl_scaled, Y_train_dl, Y_val_dl = train_test_split(
    X_scaled, Y_dl, test_size=0.2, random_state=RANDOM_SEED
)


# --- Train and Save Baseline Model ---
print("2. Training and saving Baseline Model (Logistic Regression)...")
baseline_model = LogisticRegression(random_state=RANDOM_SEED, max_iter=1000)
baseline_model.fit(X_train_scaled, Y_train_base)
joblib.dump(baseline_model, 'logistic_regression_model.joblib')


# --- Train and Save DL Model Components ---
print("3. Training Deep Learning Autoencoder...")

# Autoencoder Training
input_dim = X_train_dl_scaled.shape[1]
input_encoder = keras.Input(shape=(input_dim, ))
encoded = keras.layers.Dense(DL_ENCODING_DIM, activation=DL_ACTIVATION)(input_encoder)
decoded = keras.layers.Dense(input_dim, activation=DL_OUTPUT_ACTIVATION)(encoded)
autoencoder = keras.models.Model(input_encoder, decoded)
dl_encoder = keras.models.Model(input_encoder, encoded)

autoencoder.compile(optimizer='adam', loss='mean_squared_error')
autoencoder.fit(
    X_train_dl_scaled, X_train_dl_scaled,
    epochs=DL_PATIENCE,
    batch_size=input_dim,
    shuffle=True,
    verbose=1,
    validation_data=(X_val_dl_scaled, X_val_dl_scaled)
)

# Encode data for MLP
X_train_encoded = dl_encoder.predict(X_train_dl_scaled, verbose=0)
X_val_encoded = dl_encoder.predict(X_val_dl_scaled, verbose=0)

# MLP Training
print("4. Training Deep Learning MLP Classifier...")
inputs = keras.Input(shape=(DL_ENCODING_DIM, ))
hidden = keras.layers.Dense(DL_NEURONS, activation=DL_ACTIVATION)(inputs)
hidden = keras.layers.Dense(DL_NEURONS, activation=DL_ACTIVATION)(hidden)
bnorm = keras.layers.BatchNormalization()(hidden)
outputs = keras.layers.Dense(n_classes, activation=DL_OUTPUT_ACTIVATION)(bnorm)
dl_mlp_model = keras.Model(inputs=inputs, outputs=outputs)
optimizer = keras.optimizers.Adam(learning_rate=DL_LR)
dl_mlp_model.compile(optimizer=optimizer, loss=DL_LOSS, metrics=[DL_METRIC])

es_loss = keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=DL_PATIENCE, 
    restore_best_weights=True
)

dl_mlp_model.fit(
    X_train_encoded, Y_train_dl, 
    validation_data=(X_val_encoded, Y_val_dl), 
    batch_size=X_train_encoded.shape[0], 
    epochs=BASE_MAX_EPOCH, 
    verbose=1, 
    callbacks=[es_loss]
)

# Save DL components
print("5. Saving DL Encoder and MLP Classifier...")
dl_encoder.save('dl_encoder.keras')
dl_mlp_model.save('dl_mlp_model.keras')


# --- Save Scaler ---
print("6. Saving StandardScaler...")
joblib.dump(X_scaler, 'x_scaler.joblib')

print("\nSetup complete. You can now run the Streamlit app.")