# =====================================
# SALES FORECASTING SYSTEM
# Project: Sales Forecasting using Walmart Dataset
# Models: Random Forest Regression and XGBoost Regression
# =====================================

# =====================================
# 1. Import Libraries
# =====================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from xgboost import XGBRegressor

# =====================================
# 2. Download Dataset
# =====================================
import kagglehub

path = kagglehub.dataset_download("yasserh/walmart-dataset")

print("Dataset Path:", path)
print("Files in Dataset Folder:")
print(os.listdir(path))

# =====================================
# 3. Load Dataset
# =====================================
file_path = os.path.join(path, "Walmart.csv")
df = pd.read_csv(file_path)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nDataset Information:")
print(df.info())

# =====================================
# 4. Data Preprocessing
# =====================================

# Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

df = df.dropna()

# Duplicate Rows
print("\nDuplicate Rows:", df.duplicated().sum())
df = df.drop_duplicates()

# Convert Date Column
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

print("\nData After Cleaning:")
print(df.head())

# =====================================
# 5. Feature Engineering
# =====================================

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Week'] = df['Date'].dt.isocalendar().week.astype(int)
df['Day'] = df['Date'].dt.day
df['Quarter'] = df['Date'].dt.quarter

print("\nFeature Engineered Data:")
print(df[['Date', 'Year', 'Month', 'Week',
          'Day', 'Quarter']].head())

# =====================================
# 6. Exploratory Data Analysis
# =====================================

plt.style.use('ggplot')

# -------------------------------------
# Monthly Average Sales Trend (Clean Graph)
# -------------------------------------

monthly_sales = (
    df.groupby(['Year', 'Month'])['Weekly_Sales']
      .mean()
      .reset_index()
)

# Create labels like 2010-02, 2010-03, ...
monthly_sales['Period'] = (
    monthly_sales['Year'].astype(str) + '-' +
    monthly_sales['Month'].astype(str).str.zfill(2)
)

plt.figure(figsize=(12, 6))

plt.plot(
    monthly_sales['Period'],
    monthly_sales['Weekly_Sales'],
    marker='o',
    linewidth=2
)

plt.title('Monthly Average Sales Trend')
plt.xlabel('Month')
plt.ylabel('Average Weekly Sales')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig('monthly_sales_trend.png')
plt.show()

# =====================================
# 7. Correlation Heatmap
# =====================================

plt.figure(figsize=(12, 7))

numeric_df = df.select_dtypes(include=np.number)

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
plt.show()

# =====================================
# 8. Prepare Data for Model
# =====================================

X = df[
    [
        'Store',
        'Holiday_Flag',
        'Temperature',
        'Fuel_Price',
        'CPI',
        'Unemployment',
        'Year',
        'Month',
        'Week',
        'Day',
        'Quarter'
    ]
]

y = df['Weekly_Sales']

# =====================================
# 9. Time-Based Train-Test Split
# =====================================

# Sort dataset according to date
df = df.sort_values('Date')

X = X.loc[df.index]
y = y.loc[df.index]

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# =====================================
# 10. Random Forest Regression
# =====================================

print("\n================================")
print("RANDOM FOREST REGRESSION")
print("================================")

rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    random_state=42
)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

rf_mae = mean_absolute_error(y_test, y_pred_rf)
rf_mse = mean_squared_error(y_test, y_pred_rf)
rf_rmse = np.sqrt(rf_mse)
rf_r2 = r2_score(y_test, y_pred_rf)

print("MAE :", rf_mae)
print("MSE :", rf_mse)
print("RMSE :", rf_rmse)
print("R2 Score :", rf_r2)

# =====================================
# 11. XGBoost Regression
# =====================================

print("\n================================")
print("XGBOOST REGRESSION")
print("================================")

xgb = XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)

xgb_mae = mean_absolute_error(y_test, y_pred_xgb)
xgb_mse = mean_squared_error(y_test, y_pred_xgb)
xgb_rmse = np.sqrt(xgb_mse)
xgb_r2 = r2_score(y_test, y_pred_xgb)

print("MAE :", xgb_mae)
print("MSE :", xgb_mse)
print("RMSE :", xgb_rmse)
print("R2 Score :", xgb_r2)

# =====================================
# 12. Model Comparison
# =====================================

print("\n================================")
print("MODEL COMPARISON")
print("================================")

comparison = pd.DataFrame({
    'Model': ['Random Forest', 'XGBoost'],
    'MAE': [rf_mae, xgb_mae],
    'RMSE': [rf_rmse, xgb_rmse],
    'R2 Score': [rf_r2, xgb_r2]
})

print(comparison)

# Best Model
best_model = comparison.loc[
    comparison['MAE'].idxmin(),
    'Model'
]

print("\nBest Performing Model:", best_model)

# =====================================
# 13. Sample Prediction
# =====================================

print("\n================================")
print("SAMPLE PREDICTION")
print("================================")

sample = X_test.iloc[[0]]

print("\nInput Features:")
print(sample)

print("\nRandom Forest Prediction:")
print(rf.predict(sample))

print("\nXGBoost Prediction:")
print(xgb.predict(sample))

print("\nActual Weekly Sales:")
print(y_test.iloc[0])

# =====================================
# 14. Feature Importance
# =====================================

print("\n================================")
print("FEATURE IMPORTANCE")
print("================================")

importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
})

importance_df = importance_df.sort_values(
    by='Importance',
    ascending=False
)

print(importance_df)

plt.figure(figsize=(10, 6))

sns.barplot(
    x='Importance',
    y='Feature',
    data=importance_df
)

plt.title('Feature Importance - Random Forest')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.savefig('feature_importance.png')
plt.show()

# =====================================
# 15. Actual vs Predicted Graph
# =====================================

plt.figure(figsize=(12, 6))

plt.plot(
    y_test.values[:50],
    label='Actual Sales',
    marker='o'
)

plt.plot(
    y_pred_rf[:50],
    label='Predicted Sales (Random Forest)',
    marker='x'
)

plt.title('Actual vs Predicted Sales')
plt.xlabel('Samples')
plt.ylabel('Weekly Sales')
plt.legend()
plt.grid(True)
plt.savefig('actual_vs_predicted.png')
plt.show()

# =====================================
# 16. Future Sales Forecast
# =====================================

print("\n================================")
print("FUTURE SALES FORECAST")
print("================================")

future_sales = pd.DataFrame({
    'Actual Sales': y_test.values[:10],
    'Predicted Sales': y_pred_rf[:10]
})

print(future_sales)

plt.figure(figsize=(12, 6))

plt.plot(
    future_sales['Actual Sales'].values,
    label='Actual Sales',
    marker='o'
)

plt.plot(
    future_sales['Predicted Sales'].values,
    label='Forecasted Sales',
    marker='x'
)

plt.title('Future Sales Forecast')
plt.xlabel('Future Weeks')
plt.ylabel('Weekly Sales')
plt.legend()
plt.grid(True)
plt.savefig('future_sales_forecast.png')
plt.show()