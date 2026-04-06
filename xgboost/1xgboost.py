import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ==========================================
# SETUP & STYLING
# ==========================================
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)
colors = sns.color_palette("viridis", 8)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# ==========================================
# DATA LOADING
# ==========================================
print("Loading dataset...")
# Make sure path matches your generated file
df = pd.read_csv('simulated_lorawan_data/synthetic_lorawan_25_fields.csv', parse_dates=['timestamp'])

example_field = df['field_id'].iloc[0]

# ==========================================
# PHASE 0: FEATURE DISTRIBUTIONS & OVERVIEW
# ==========================================
print("\n--- Phase 0: Initial Feature Distributions (Hourly Sensor Data) ---")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.histplot(df['air_temperature_c'], kde=True, ax=axes[0, 0], color='coral')
axes[0, 0].set_title('Air Temperature Distribution (°C)', fontsize=14)

sns.histplot(df['par_umol_m2_s'], kde=True, ax=axes[0, 1], color='orange')
axes[0, 1].set_title('PAR / Solar Radiation Distribution', fontsize=14)

sns.histplot(df['soil_ph'], kde=True, ax=axes[1, 0], color='lightgreen')
axes[1, 0].set_title('Soil pH Distribution', fontsize=14)

sns.histplot(df['soil_moisture_pct'], kde=True, ax=axes[1, 1], color='teal')
axes[1, 1].set_title('Soil Moisture (%) Distribution', fontsize=14)

fig.tight_layout()
plt.show()

# ==========================================
# PHASE 1: DATA QUALITY & ANOMALY DETECTION
# ==========================================
print("\n--- Phase 1: Data Quality (Time-Series Outliers) ---")

iso_forest = IsolationForest(contamination=0.01, random_state=42)

# Run outlier detection on a subset (one field) for memory efficiency
df_field = df[df['field_id'] == example_field].copy()
df_field['Is_Outlier'] = iso_forest.fit_predict(df_field[['air_temperature_c']])

# Clean impossible temps globally
df.loc[df['air_temperature_c'] > 60, 'air_temperature_c'] = np.nan

# ==========================================
# PHASE 2: AGGREGATION & TARGET GENERATION
# ==========================================
print("\n--- Phase 2: Aggregating to Season-Level & Generating Target ---")

# Compress hourly data into a single row per field to predict yield
numeric_cols = df.select_dtypes(include=[np.number]).columns
df_agg = df.groupby(['field_id', 'crop_type'])[numeric_cols].mean().reset_index()

# Add suffixes to indicate these are aggregated averages
df_agg.columns = ['field_id', 'crop_type'] + [f"{col}_mean" for col in numeric_cols]

# Generate synthetic yield
npk_bonus = (df_agg['nitrogen_mg_kg_mean'] * 5) + (df_agg['phosphorus_mg_kg_mean'] * 3)
temp_penalty = abs(df_agg['air_temperature_c_mean'] - 22) * 150 
moisture_bonus = df_agg['soil_moisture_pct_mean'] * 20

df_agg['yield_kg_per_hectare'] = 3000 + npk_bonus + moisture_bonus - temp_penalty + np.random.normal(0, 200, len(df_agg))

plt.figure(figsize=(8, 5))
sns.histplot(df_agg['yield_kg_per_hectare'], kde=True, color=colors[4], bins=10)
plt.title("Distribution of Final Field Yields", fontsize=14)
plt.xlabel("Yield (kg/ha)")
plt.axvline(df_agg['yield_kg_per_hectare'].mean(), color='red', linestyle='--', label='Mean Yield')
plt.legend()
plt.show()

# ==========================================
# PHASE 3 & 4: UNI/BIVARIATE & MULTICOLLINEARITY (VIF)
# ==========================================
print("\n--- Phase 3 & 4: Feature Relationships & Multicollinearity ---")

# Replace NDVI check with PAR (Radiation) check
plt.figure(figsize=(8, 5))
sns.regplot(
    x='par_umol_m2_s_mean', 
    y='yield_kg_per_hectare', 
    data=df_agg, 
    scatter_kws={'alpha':0.8, 'color': colors[0], 's': 100}, 
    line_kws={'color':'red', 'linewidth': 2}
)
plt.title("Average Season Radiation (PAR) vs. Yield", fontsize=14)
plt.xlabel("PAR Mean")
plt.ylabel("Yield (kg/ha)")
plt.tight_layout()
plt.show()

print("\nCalculating Variance Inflation Factor (VIF)...")

# Prepare features for VIF (excluding IDs and Target)
vif_features = [col for col in df_agg.columns if col not in ['field_id', 'crop_type', 'yield_kg_per_hectare']]
df_vif = df_agg[vif_features].dropna() 

iterations = 3

for i in range(iterations):
    print(f"\n--- VIF Drop Iteration {i+1} ---")
    vif_data = pd.DataFrame()
    vif_data["Feature"] = df_vif.columns
    # Add constant to avoid perfect multicollinearity errors in statsmodels
    vif_data["VIF"] = [variance_inflation_factor(df_vif.values, j) for j in range(len(df_vif.columns))]
    vif_data = vif_data.sort_values(by="VIF", ascending=False).reset_index(drop=True)
    print(vif_data.head(3))
    
    worst_feature = vif_data.iloc[0]["Feature"]
    worst_vif_score = vif_data.iloc[0]["VIF"]
    
    print(f"\n>> Dropping '{worst_feature}' (VIF: {worst_vif_score:.2f})")
    df_vif = df_vif.drop(columns=[worst_feature])
    df_agg = df_agg.drop(columns=[worst_feature], errors='ignore')

# Final VIF Check
print("\n--- Final VIF Scores (After Drops) ---")
final_vif_data = pd.DataFrame()
final_vif_data["Feature"] = df_vif.columns
final_vif_data["VIF"] = [variance_inflation_factor(df_vif.values, j) for j in range(len(df_vif.columns))]
final_vif_data = final_vif_data.sort_values(by="VIF", ascending=False).reset_index(drop=True)

plt.figure(figsize=(8, 5))
sns.barplot(x='VIF', y='Feature', data=final_vif_data, palette='magma')
plt.axvline(x=5, color='red', linestyle='--', label='Threshold (5)')
plt.title(f"Variance Inflation Factor (After {iterations} Drops)", fontsize=14)
plt.legend()
plt.tight_layout()
plt.show()

# ==========================================
# PHASE 5: CROSS-SECTIONAL ANALYSIS
# ==========================================
print("\n--- Phase 5: Cross-Sectional Analysis ---")

plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='nitrogen_mg_kg_mean', 
    y='yield_kg_per_hectare', 
    hue='crop_type', 
    data=df_agg, 
    palette='Set2', 
    alpha=0.9,
    s=150
)
plt.title("Average Season Nitrogen vs. Crop Yield", fontsize=14)
plt.xlabel("Nitrogen Mean (mg/kg)")
plt.ylabel("Yield (kg/ha)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Crop Type") 
plt.tight_layout()
plt.show()

# ==========================================
# PHASE 6: INITIAL FEATURE IMPORTANCE
# ==========================================
print("\n--- Phase 6: Base Feature Importance (All Crops) ---")

# Skip crop filtering due to small sample size (25 fields). Train on all.
df_modeling = df_agg.drop(columns=['field_id']).dropna()
df_encoded = pd.get_dummies(df_modeling, drop_first=True)

X = df_encoded.drop(columns=['yield_kg_per_hectare'])
y = df_encoded['yield_kg_per_hectare']

rf = RandomForestRegressor(n_estimators=1000, random_state=42, n_jobs=-1)
rf.fit(X, y)

importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(10), palette='viridis')
plt.title("Top Predictors for Yield (Random Forest)", fontsize=14)
plt.xlabel("Relative Importance")
plt.tight_layout()
plt.show()

##############################################################################
######################   XGBoost training   ###################################
##############################################################################

# ==========================================
# PHASE 7: XGBOOST & HYPERPARAMETER TUNING
# ==========================================
print("\n--- Phase 7: XGBoost Model Training & Validation ---")

X_xgb = X
y_xgb = y

# 2. Perform Split (Modified ratios for small dataset)
X_temp, X_test, y_temp, y_test = train_test_split(X_xgb, y_xgb, test_size=0.20, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.20, random_state=42)

print(f"\nData Shapes:")
print(f"Train:      {X_train.shape}")
print(f"Validation: {X_val.shape}")
print(f"Test:       {X_test.shape}")

# 3. Setup XGBoost
xgb_estimator = xgb.XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
    early_stopping_rounds=15, # Reduced early stopping patience for small data
    eval_metric='rmse'
)

param_grid = {
    'n_estimators': [50, 100],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5],      
}

# 5. Execute GridSearchCV
print("\nInitiating GridSearchCV...")
grid_search = GridSearchCV(
    estimator=xgb_estimator,
    param_grid=param_grid,
    cv=3, # Reduced to 3 folds because we only have 25 fields total!
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(
    X_train, 
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False 
)

print(f"\nBest Hyperparameters Found:")
for param, value in grid_search.best_params_.items():
    print(f" - {param}: {value}")

best_xgb = grid_search.best_estimator_

# ==========================================
# PHASE 8: MODEL EVALUATION
# ==========================================
print("\n--- Phase 8: Final Evaluation ---")

y_val_pred = best_xgb.predict(X_val)
y_test_pred = best_xgb.predict(X_test)

def print_metrics(y_true, y_pred, dataset_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{dataset_name} Metrics:")
    print(f"  RMSE: {rmse:.2f} kg/ha")
    print(f"  MAE:  {mae:.2f} kg/ha")
    print(f"  R²:   {r2:.4f}\n")

print_metrics(y_val, y_val_pred, "Validation Set")
print_metrics(y_test, y_test_pred, "Holdout Test Set")

# ==========================================
# PHASE 8A: ADVANCED MODEL DIAGNOSTICS
# ==========================================
print("\n--- Generating Diagnostic Plots ---")
residuals = y_test - y_test_pred

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Actual vs Predicted
sns.scatterplot(x=y_test, y=y_test_pred, alpha=0.9, color=colors[3], s=100, ax=axes[0])
min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())
axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction (1:1)')
axes[0].set_title("Actual vs. Predicted Yield", fontsize=14, fontweight='bold')
axes[0].set_xlabel("Actual Yield (kg/ha)")
axes[0].set_ylabel("Predicted Yield (kg/ha)")
axes[0].legend()

# Residuals
sns.scatterplot(x=y_test_pred, y=residuals, alpha=0.9, color=colors[5], s=100, ax=axes[1])
axes[1].axhline(0, color='r', linestyle='--', lw=2)
axes[1].set_title("Residuals vs. Predicted Values", fontsize=14, fontweight='bold')
axes[1].set_xlabel("Predicted Yield (kg/ha)")
axes[1].set_ylabel("Error (Residuals in kg/ha)")

fig.tight_layout(pad=3.0)
plt.show()

# ==========================================
# PHASE 9: SAVING THE MODEL FOR PRODUCTION
# ==========================================
print("\n--- Phase 9: Model Export ---")

os.makedirs('model_artifacts', exist_ok=True)

model_path = 'model_artifacts/xgboost_lorawan_yield_model.joblib'
joblib.dump(best_xgb, model_path)
print(f"[*] Model successfully saved to: {model_path}")

features_path = 'model_artifacts/model_features_lorawan.joblib'
joblib.dump(list(X_xgb.columns), features_path)
print(f"[*] Feature schema successfully saved to: {features_path}")

print("\nReady for deployment!")