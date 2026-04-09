import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import os

# Define the exact GCS path where Vertex AI saved your artifacts
MODEL_DIR = "gs://your-project-ml-data/models/v1/"

print(f"[*] Loading Artifacts from {MODEL_DIR}...")

# Load Model natively from GCS
model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'crop_lstm_model.h5'))

# Use tf.io.gfile to read numpy arrays and pickles from GCS
with tf.io.gfile.GFile(os.path.join(MODEL_DIR, 'X_test.npy'), 'rb') as f:
    X_test = np.load(f)

with tf.io.gfile.GFile(os.path.join(MODEL_DIR, 'y_test.npy'), 'rb') as f:
    y_test = np.load(f)

with tf.io.gfile.GFile(os.path.join(MODEL_DIR, 'training_history.pkl'), 'rb') as f:
    history_dict = pickle.load(f)

# ==========================================
# EVALUATION & VISUALIZATION
# ==========================================
print("\n--- Evaluating LSTM ---")
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss (MSE): {loss:.4f}")
print(f"Test MAE: {mae:.4f}")

y_pred = model.predict(X_test)

plt.figure(figsize=(15, 5))

# Plot 1: Loss
plt.subplot(1, 2, 1)
plt.plot(history_dict['loss'], label='Train Loss')
plt.plot(history_dict['val_loss'], label='Val Loss')
plt.title('Model Loss (MSE)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot 2: Predictions
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred, alpha=0.5, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('LSTM: Actual vs Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')

plt.tight_layout()
plt.show()