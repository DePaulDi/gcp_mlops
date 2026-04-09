import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import os
import gc
import pickle

# ==========================================
# 1. DATA PREPARATION
# ==========================================
print("[*] Loading and Preprocessing Sequence Data from GCS...")
data_path = 'gs://lorawan_simulated_data/data_v1/synthetic_lorawan_25_fields.csv'
df = pd.read_csv(data_path)

floats = df.select_dtypes(include=['float64']).columns
df[floats] = df[floats].astype('float32')

features = [
    'soil_moisture_pct', 'air_moisture_rh', 'soil_temperature_c', 
    'air_temperature_c', 'soil_ph', 'soil_ec_ds_m', 'nitrogen_mg_kg', 
    'phosphorus_mg_kg', 'potassium_mg_kg', 'par_umol_m2_s', 'uv_index'
]

scaler = RobustScaler()
df[features] = scaler.fit_transform(df[features])

def create_sequences(data, window_size=24):
    X, y = [], []
    for field in data['field_id'].unique():
        field_data = data[data['field_id'] == field][features].values
        field_yield = (data[data['field_id'] == field]['nitrogen_mg_kg'] * 0.5).values 
        
        for i in range(window_size, len(field_data), 24): 
            X.append(field_data[i-window_size:i])
            y.append(field_yield[i])
            
    return np.array(X), np.array(y)

WINDOW_SIZE = 168 * 4 * 2
X, y = create_sequences(df, window_size=WINDOW_SIZE)

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

del df
gc.collect()

# ==========================================
# 2. SAVE TEST ARTIFACTS TO GCS
# ==========================================
# Vertex AI populates AIP_MODEL_DIR. If running locally, default to a local/GCS test path.
model_dir = os.getenv("AIP_MODEL_DIR", "gs://your-project-ml-data/models/v1/")
print(f"[*] Artifacts will be saved to: {model_dir}")

# We use tf.io.gfile to write directly to GCS buckets
with tf.io.gfile.GFile(os.path.join(model_dir, 'X_test.npy'), 'wb') as f:
    np.save(f, X_test)
with tf.io.gfile.GFile(os.path.join(model_dir, 'y_test.npy'), 'wb') as f:
    np.save(f, y_test)

# ==========================================
# 3. LSTM ARCHITECTURE & TRAINING
# ==========================================
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    BatchNormalization(),
    Dropout(0.2),
    LSTM(128, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

print("\n--- Training LSTM Model ---")
history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=256,
    validation_data=(X_val, y_val),
    verbose=1
)

# ==========================================
# 4. EXPORT MODEL & HISTORY TO GCS
# ==========================================
model.save(os.path.join(model_dir, 'crop_lstm_model.h5'))

with tf.io.gfile.GFile(os.path.join(model_dir, 'training_history.pkl'), 'wb') as f:
    pickle.dump(history.history, f)

print(f"[*] Job complete. All artifacts saved to {model_dir}")