import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV

# 데이터 불러오기
data = pd.read_csv('data/df.csv', encoding='cp949')
data.dropna(inplace=True)
# 필요한 변수만 추출
X = data[['기온(°C)', '누적강수량(mm)', '현지기압(hPa)', '해면기압(hPa)', '일조(Sec)']]
y = data['선량률(nSv/h)']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Define the parameter grid for RandomizedSearchCV
param_dist = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5, 6],
    'min_child_weight': [1, 2, 3],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'gamma': [0, 1, 5],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [0, 0.1, 1]
}

# Create the XGBoost regressor
xgb = XGBRegressor(objective='reg:squarederror')

# Create RandomizedSearchCV object
random_search = RandomizedSearchCV(xgb, param_distributions=param_dist, n_iter=10, scoring='neg_mean_squared_error', cv=3, n_jobs=-1)

# 학습 데이터와 검증 데이터 준비
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.3, random_state=511)

# Fit the model with RandomizedSearchCV
random_search.fit(X_train, y_train)

# Get the best estimator
best_xgboost_model = random_search.best_estimator_

# Print the best hyperparameters
print("Best hyperparameters:", random_search.best_params_)

# Train the best model on the training data
best_xgboost_model.fit(X_train, y_train)

# Use the model on the validation data
y_val_pred = best_xgboost_model.predict(X_val)

# RMSE and R2 scores for the best model
rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
r2 = r2_score(y_val, y_val_pred)
print("Validation RMSE (Best Model):", rmse)
print("Validation R2 Score (Best Model):", r2)

# Now, let's train a model without hyperparameter tuning for comparison
# Create an XGBoost regressor
xgb_without_tuning = XGBRegressor(objective='reg:squarederror', random_state=42)

# Train the untuned model on the training data
xgb_without_tuning.fit(X_train, y_train)

# Use the untuned model on the validation data
y_val_pred_no_tuning = xgb_without_tuning.predict(X_val)

# RMSE and R2 scores for the untuned model
rmse_no_tuning = np.sqrt(mean_squared_error(y_val, y_val_pred_no_tuning))
r2_no_tuning = r2_score(y_val, y_val_pred_no_tuning)
print("Validation RMSE (Untuned Model):", rmse_no_tuning)
print("Validation R2 Score (Untuned Model):", r2_no_tuning)

# Best hyperparameters: {'subsample': 0.9, 'reg_lambda': 0.1, 'reg_alpha': 0, 'n_estimators': 300, 'min_child_weight': 1, 'max_depth': 6, 'learning_rate': 0.2, 'gamma': 1, 'colsample_bytree': 0.8}
# Validation RMSE (Best Model): 21.223521664362796
# Validation R2 Score (Best Model): 0.7684196161342411
# Validation RMSE (Untuned Model): 22.92849798691727
# Validation R2 Score (Untuned Model): 0.7297174031915172