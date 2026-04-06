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

# Suppress warnings
warnings.filterwarnings('ignore')

# ==========================================
# SETUP & STYLING
# ==========================================
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)
colors = sns.color_palette("viridis", 8)

# ==========================================
# DATA LOADING
# ==========================================
print("Loading dataset...")
df = pd.read_csv('simulated_lorawan_data/synthetic_lorawan_25_fields.csv', parse_dates=['timestamp'])

# ==========================================
# PHASE 1: DATA QUALITY
# ==========================================
# Clean impossible temps globally
df.loc[df['air_temperature_c'] > 60, 'air_temperature_c'] = np.nan

# ==========================================
# PHASE 2: MONTHLY AGGREGATION & TARGET GENERATION
# ==========================================
print("\n--- Phase 2: Aggregating to Monthly Snapshots ---")

# 1. Ensure timestamp is the index for resampling
df_time = df.set_index('timestamp')

# 2. Group by Field, Crop, and Month
# Note: df_monthly is now our main dataframe for ML
df_monthly = df_time.groupby(['field_id', 'crop_type', pd.Grouper(freq='MS')]).mean().reset_index()

# 3. Rename columns to reflect monthly averages
numeric_cols = df.select_dtypes(include=[np.number]).columns
df_monthly.columns = ['field_id', 'crop_type', 'month_timestamp'] + [f"{col}_mean" for col in numeric_cols]

# 4. Generate Synthetic Yield
month_val = df_monthly['month_timestamp'].dt.month
seasonal_factor = np.sin((month_val - 1) * np.pi / 6)

npk_bonus = (df_monthly['nitrogen_mg_kg_mean'] * 5) + (df_monthly['phosphorus_mg_kg_mean'] * 3)
temp_penalty = abs(df_monthly['air_temperature_c_mean'] - 22) * 150 

df_monthly['yield_kg_per_hectare'] = (3000 + npk_bonus - temp_penalty) * (0.5 + 0.5 * seasonal_factor)
df_monthly['yield_kg_per_hectare'] += np.random.normal(0, 150, len(df_monthly))

print(f"New Dataset Shape: {df_monthly.shape}")

# ==========================================
# PHASE 3 & 4: RELATIONSHIPS & VIF (Fixed Variable Names)
# ==========================================
print("\n--- Phase 3 & 4: Feature Relationships & Multicollinearity ---")

plt.figure(figsize=(8, 5))
sns.regplot(
    x='par_umol_m2_s_mean', 
    y='yield_kg_per_hectare', 
    data=df_monthly, # Fixed from df_agg
    scatter_kws={'alpha':0.6, 'color': colors[0], 's': 30}, 
    line_kws={'color':'red', 'linewidth': 2}
)
plt.title("Monthly Radiation (PAR) vs. Predicted Yield", fontsize=14)
plt.show()

print("\nCalculating Variance Inflation Factor (VIF)...")

# Prepare features for VIF
vif_features = [col for col in df_monthly.columns if col not in ['field_id', 'crop_type', 'yield_kg_per_hectare', 'month_timestamp']]
df_vif = df_monthly[vif_features].dropna() 

iterations = 3
for i in range(iterations):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = df_vif.columns
    vif_data["VIF"] = [variance_inflation_factor(df_vif.values, j) for j in range(len(df_vif.columns))]
    vif_data = vif_data.sort_values(by="VIF", ascending=False).reset_index(drop=True)
    
    worst_feature = vif_data.iloc[0]["Feature"]
    print(f"Iteration {i+1}: Dropping '{worst_feature}' (VIF: {vif_data.iloc[0]['VIF']:.2f})")
    df_vif = df_vif.drop(columns=[worst_feature])
    df_monthly = df_monthly.drop(columns=[worst_feature], errors='ignore') # Fixed from df_agg

# ==========================================
# PHASE 6: INITIAL FEATURE IMPORTANCE
# ==========================================
print("\n--- Phase 6: Base Feature Importance ---")

# Drop non-predictive columns
df_modeling = df_monthly.drop(columns=['field_id', 'month_timestamp']).dropna()
df_encoded = pd.get_dummies(df_modeling, drop_first=True)

X = df_encoded.drop(columns=['yield_kg_per_hectare'])
y = df_encoded['yield_kg_per_hectare']

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X, y)

# ==========================================
# PHASE 7: XGBOOST & HYPERPARAMETER TUNING
# ==========================================
print("\n--- Phase 7: XGBoost Model Training ---")

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

print(f"Train size: {X_train.shape[0]} | Val size: {X_val.shape[0]} | Test size: {X_test.shape[0]}")

xgb_estimator = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5]
}

grid_search = GridSearchCV(
    estimator=xgb_estimator,
    param_grid=param_grid,
    cv=5, # Now we have enough rows for 5-fold CV!
    scoring='neg_root_mean_squared_error',
    n_jobs=-1
)

grid_search.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
best_xgb = grid_search.best_estimator_

# ==========================================
# PHASE 8: EVALUATION
# ==========================================
y_test_pred = best_xgb.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2 = r2_score(y_test, y_test_pred)

print(f"\nFinal Test Metrics:\nRMSE: {rmse:.2f} kg/ha\nR2 Score: {r2:.4f}")

# ==========================================
# PHASE 9: EXPORT
# ==========================================
os.makedirs('model_artifacts', exist_ok=True)
joblib.dump(best_xgb, 'model_artifacts/xgboost_lorawan_monthly.joblib')
print("\n[*] Model saved. Ready for MLOps pipeline.")