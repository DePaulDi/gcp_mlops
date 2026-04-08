import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import gc

# ==========================================
# 1. DATA PREPARATION (WINDOWING)
# ==========================================
print("Loading and Preprocessing Sequence Data...")
# df = pd.read_csv('../simulated_lorawan_data/synthetic_lorawan_25_fields.csv')
df = pd.read_csv('gs://my-first-project/simulated_lorawan_data/synthetic_lorawan_25_fields.csv')

# Memory optimization: Downcast to float32
floats = df.select_dtypes(include=['float64']).columns
df[floats] = df[floats].astype('float32')

# Define features (Exclude IDs and timestamps)
features = [
    'soil_moisture_pct', 'air_moisture_rh', 'soil_temperature_c', 
    'air_temperature_c', 'soil_ph', 'soil_ec_ds_m', 'nitrogen_mg_kg', 
    'phosphorus_mg_kg', 'potassium_mg_kg', 'par_umol_m2_s', 'uv_index'
]

# Standardize features
scaler = RobustScaler()
df[features] = scaler.fit_transform(df[features])

def create_sequences(data, window_size=24):
    """
    Transforms flat data into (samples, window_size, n_features)
    window_size=24 means the model looks at the last 24 hours to predict.
    """
    X, y = [], []
    # We group by field so sequences don't 'bleed' from one farm to another
    for field in data['field_id'].unique():
        field_data = data[data['field_id'] == field][features].values
        
        # For synthetic target: we use the same logic as before but at each timestep
        # In a real scenario, 'y' would be the final recorded harvest yield
        field_yield = (data[data['field_id'] == field]['nitrogen_mg_kg'] * 0.5).values 
        
        for i in range(window_size, len(field_data), 24): # Step by 24 to save memory
            X.append(field_data[i-window_size:i])
            y.append(field_yield[i])
            
    return np.array(X), np.array(y)

# We use a 168-hour window (1 week of hourly data)
WINDOW_SIZE = 168 
X, y = create_sequences(df, window_size=WINDOW_SIZE)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Shape of X_train: {X_train.shape}") # (Samples, Time Steps, Features)
del df
gc.collect()

# ==========================================
# 2. LSTM ARCHITECTURE
# ==========================================

model = Sequential([
    # First LSTM Layer
    LSTM(32, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    BatchNormalization(),
    Dropout(0.2),
    
    # Second LSTM Layer
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    
    # Fully Connected Layers
    Dense(16, activation='relu'),
    Dense(1) # Regression output
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# ==========================================
# 3. TRAINING
# ==========================================
print("\n--- Training LSTM Model ---")
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# ==========================================
# 4. EVALUATION & VISUALIZATION
# ==========================================
print("\n--- Evaluating LSTM ---")
y_pred = model.predict(X_test)

plt.figure(figsize=(15, 5))

# Plot 1: Training Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss (MSE)')
plt.legend()

# Plot 2: Actual vs Predicted
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred, alpha=0.5, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('LSTM: Actual vs Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')

plt.tight_layout()
plt.show()

# Save Model
os.makedirs('model_artifacts', exist_ok=True)
# Save Model directly to GCS
model.save('gs://your-project-ml-data/models/v1/crop_lstm_model.h5')
print("[*] LSTM Model saved to model_artifacts/crop_lstm_model.h5")
