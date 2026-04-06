import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import gc
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. LOAD & DOWNCAST (The Memory Saver)
# ==========================================
print("Loading dataset...")
# Using float32 instead of float64 saves 50% memory immediately
df = pd.read_csv('simulated_lorawan_data/synthetic_lorawan_25_fields.csv', parse_dates=['timestamp'])

# Identify float columns and downcast them
floats = df.select_dtypes(include=['float64']).columns
df[floats] = df[floats].astype('float32')

print(f"[*] Raw data loaded. Shape: {df.shape}")

# ==========================================
# 2. EFFICIENT AGGREGATION (Monthly Snapshots)
# ==========================================
print("\n--- Phase 2: Aggregating to Monthly Snapshots (Optimized) ---")

# Set index to timestamp to use the high-performance .resample() method
df = df.set_index('timestamp')

# We group by field and crop, then resample the time-series to Month Start (MS)
# We calculate the mean for all numeric columns
df_monthly = df.groupby(['field_id', 'crop_type']).resample('MS').mean().drop(columns=['field_id', 'crop_type'], errors='ignore').reset_index()

# Standardize the timestamp column name
df_monthly = df_monthly.rename(columns={'timestamp': 'month_timestamp'})

# --- CLEANUP RAW DATA TO FREE RAM ---
del df
gc.collect() 
print(f"[*] Aggregation complete. Raw data cleared from RAM. New Shape: {df_monthly.shape}")

# ==========================================
# 3. TARGET GENERATION (Synthetic Yield)
# ==========================================
# We use the monthly averages to create a potential yield value
month_val = df_monthly['month_timestamp'].dt.month
seasonal_factor = np.sin((month_val - 1) * np.pi / 6)

# Access columns (handle naming differences if they exist)
n = df_monthly['nitrogen_mg_kg']
p = df_monthly['phosphorus_mg_kg']
temp = df_monthly['air_temperature_c']
moist = df_monthly['soil_moisture_pct']

# Logic: Good NPK + Ideal Temp (22C) + Moisture + Seasonality = High Yield
yield_val = (3000 + (n*5) + (p*3) + (moist*20) - abs(temp-22)*150) * (0.5 + 0.5 * seasonal_factor)
df_monthly['yield_kg_per_hectare'] = yield_val + np.random.normal(0, 100, len(df_monthly))

# ==========================================
# 4. MULTICOLLINEARITY (VIF)
# ==========================================
print("\n--- Phase 3 & 4: VIF Feature Selection ---")

# Select only numeric features for VIF, excluding the target and time
vif_features = [col for col in df_monthly.select_dtypes(include=[np.number]).columns 
                if col != 'yield_kg_per_hectare']

df_vif = df_monthly[vif_features].dropna()

# Drop top 3 redundant features
for i in range(3):
    vif_series = pd.Series([variance_inflation_factor(df_vif.values, j) 
                           for j in range(df_vif.shape[1])], index=df_vif.columns)
    worst = vif_series.sort_values(ascending=False).index[0]
    print(f"Iteration {i+1}: Dropping '{worst}' (VIF: {vif_series[worst]:.2f})")
    df_vif = df_vif.drop(columns=[worst])
    df_monthly = df_monthly.drop(columns=[worst], errors='ignore')

# ==========================================
# 5. MODEL PREPARATION
# ==========================================
# Create Dummy Variables for 'crop_type'
df_modeling = df_monthly.drop(columns=['field_id', 'month_timestamp']).dropna()
df_encoded = pd.get_dummies(df_modeling, drop_first=True)

X = df_encoded.drop(columns=['yield_kg_per_hectare'])
y = df_encoded['yield_kg_per_hectare']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Split train into train and validation (for XGBoost early stopping)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# ==========================================
# 6. XGBOOST TRAINING (The MLOps Part)
# ==========================================
print("\n--- Phase 7: Training XGBoost with GridSearchCV ---")

xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5],
    'subsample': [0.8]
}

grid = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='r2')
grid.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

best_model = grid.best_estimator_
print(f"Best Params: {grid.best_params_}")

# ==========================================
# 7. EVALUATION & EXPORT
# ==========================================
y_pred = best_model.predict(X_test)
print(f"\nFinal Results:")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")

os.makedirs('model_artifacts', exist_ok=True)
joblib.dump(best_model, 'model_artifacts/optimized_xgboost_model.joblib')
print("\n[*] Success! Model saved in 'model_artifacts/'.")

# ==========================================
# 8. MODEL DIAGNOSTICS & VISUALIZATION
# ==========================================
print("\n--- Phase 8: Generating Diagnostic Plots ---")

# --- RE-DEFINE COLORS AND STYLE ---
sns.set_style("whitegrid")
colors = sns.color_palette("viridis", 10)
# ----------------------------------

# Generate predictions for the test set
y_test_pred = best_model.predict(X_test)
residuals = y_test - y_test_pred

# Create a 2x2 visualization grid
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Actual vs. Predicted (The 45-degree Line)
sns.scatterplot(x=y_test, y=y_test_pred, alpha=0.6, color=colors[3], s=80, ax=axes[0, 0])
line_min = min(y_test.min(), y_test_pred.min())
line_max = max(y_test.max(), y_test_pred.max())
axes[0, 0].plot([line_min, line_max], [line_min, line_max], 'r--', lw=2, label='Perfect Prediction')
axes[0, 0].set_title("Actual vs. Predicted Yield", fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel("Actual Yield (kg/ha)")
axes[0, 0].set_ylabel("Predicted Yield (kg/ha)")
axes[0, 0].legend()

# 2. Residuals vs. Predicted (Error Consistency)
sns.scatterplot(x=y_test_pred, y=residuals, alpha=0.5, color=colors[5], s=80, ax=axes[0, 1])
axes[0, 1].axhline(0, color='r', linestyle='--', lw=2)
axes[0, 1].set_title("Residuals vs. Predicted Values", fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel("Predicted Yield (kg/ha)")
axes[0, 1].set_ylabel("Error (Residuals)")

# 3. Distribution of Errors (Normality Check)
sns.histplot(residuals, kde=True, color=colors[2], ax=axes[1, 0])
axes[1, 0].axvline(0, color='r', linestyle='--', lw=2)
axes[1, 0].set_title("Distribution of Prediction Errors", fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel("Residual (kg/ha)")

# 4. Feature Importance (XGBoost Internal Weighting)
xgb_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

sns.barplot(x='Importance', y='Feature', data=xgb_importance.head(10), palette='magma', ax=axes[1, 1])
axes[1, 1].set_title("Top 10 Drivers of Crop Yield", fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

# ==========================================
# EXTRA: ERROR ANALYSIS BY CROP TYPE
# ==========================================
# This helps see if the model is failing on one specific crop (e.g. Rice vs Corn)
test_results = X_test.copy()
test_results['Actual'] = y_test
test_results['Predicted'] = y_test_pred
test_results['Abs_Error'] = np.abs(residuals)

# Find crop columns (since they were One-Hot Encoded)
crop_cols = [col for col in test_results.columns if 'crop_type_' in col]

if crop_cols:
    # Reverse One-Hot Encoding for a quick plot
    test_results['Crop'] = test_results[crop_cols].idxmax(axis=1).str.replace('crop_type_', '')
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Crop', y='Abs_Error', data=test_results, palette='Set2')
    plt.title("Absolute Prediction Error by Crop Type", fontsize=14, fontweight='bold')
    plt.ylabel("Absolute Error (kg/ha)")
    plt.show()

print(f"[*] Mean Absolute Error (MAE): {np.mean(np.abs(residuals)):.2f} kg/ha")