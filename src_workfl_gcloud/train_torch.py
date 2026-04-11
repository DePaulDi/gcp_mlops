import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import os
import gc
import pickle
import fsspec

# ==========================================
# 0. HARDWARE SETUP
# ==========================================
# Explicitly target the GPU if it is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Using device: {device}")

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

# Convert NumPy arrays to PyTorch Tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1) # Match output shape (batch, 1)

X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

# Create DataLoaders for batching
batch_size = 256
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)

# ==========================================
# 2. SAVE TEST ARTIFACTS TO GCS
# ==========================================
model_dir = os.getenv("AIP_MODEL_DIR", "gs://your-project-ml-data/models/v1/")
print(f"[*] Artifacts will be saved to: {model_dir}")

# Using fsspec to handle GCS writes natively like tf.io.gfile
with fsspec.open(os.path.join(model_dir, 'X_test.npy'), 'wb') as f:
    np.save(f, X_test)
with fsspec.open(os.path.join(model_dir, 'y_test.npy'), 'wb') as f:
    np.save(f, y_test)

# ==========================================
# 3. LSTM ARCHITECTURE
# ==========================================
class CropYieldLSTM(nn.Module):
    def __init__(self, input_dim):
        super(CropYieldLSTM, self).__init__()
        # First LSTM layer (equivalent to return_sequences=True)
        self.lstm1 = nn.LSTM(input_dim, 128, batch_first=True)
        
        # PyTorch BatchNorm1d expects (batch_size, channels, sequence_length) 
        # so we will permute the dimensions in the forward pass.
        self.bn = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.2)
        
        # Second LSTM layer
        self.lstm2 = nn.LSTM(128, 128, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        
        # Dense layers
        self.fc1 = nn.Linear(128, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x, _ = self.lstm1(x)
        
        # Permute for BatchNorm: (batch, features, seq_len)
        x = x.permute(0, 2, 1)
        x = self.bn(x)
        x = x.permute(0, 2, 1) # Back to (batch, seq_len, features)
        
        x = self.dropout1(x)
        
        # Second LSTM (equivalent to return_sequences=False)
        x, _ = self.lstm2(x)
        x = x[:, -1, :] # Grab the hidden state of the last time step
        
        x = self.dropout2(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CropYieldLSTM(input_dim=X_train.shape[2]).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# ==========================================
# 4. TRAINING LOOP
# ==========================================
print("\n--- Training PyTorch LSTM Model ---")
epochs = 5
history = {'loss': [], 'mae': [], 'val_loss': [], 'val_mae': []}

for epoch in range(epochs):
    model.train()
    train_loss, train_mae = 0.0, 0.0
    
    for X_batch, y_batch in train_loader:
        # Move data to GPU
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * X_batch.size(0)
        train_mae += torch.abs(outputs - y_batch).sum().item()
        
    train_loss /= len(train_loader.dataset)
    train_mae /= len(train_loader.dataset)
    
    # Validation phase
    model.eval()
    val_loss, val_mae = 0.0, 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            val_loss += loss.item() * X_batch.size(0)
            val_mae += torch.abs(outputs - y_batch).sum().item()
            
    val_loss /= len(val_loader.dataset)
    val_mae /= len(val_loader.dataset)
    
    # Track metrics
    history['loss'].append(train_loss)
    history['mae'].append(train_mae)
    history['val_loss'].append(val_loss)
    history['val_mae'].append(val_mae)
    
    print(f"Epoch {epoch+1}/{epochs} - loss: {train_loss:.4f} - mae: {train_mae:.4f} - val_loss: {val_loss:.4f} - val_mae: {val_mae:.4f}")

# ==========================================
# 5. EXPORT MODEL & HISTORY TO GCS
# ==========================================
# PyTorch allows direct saving to file-like objects, so we use fsspec again.
with fsspec.open(os.path.join(model_dir, 'crop_lstm_model.pth'), 'wb') as f:
    torch.save(model.state_dict(), f)

with fsspec.open(os.path.join(model_dir, 'training_history.pkl'), 'wb') as f:
    pickle.dump(history, f)

print(f"[*] Job complete. All artifacts saved to {model_dir}")