import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

# 데이터 불러오기
data = pd.read_csv('data/df.csv', encoding='cp949')
data.dropna(inplace=True)

# 필요한 변수만 추출
X = data[['기온(°C)', '누적강수량(mm)', '현지기압(hPa)', '해면기압(hPa)', '일조(Sec)']]
y = data['선량률(nSv/h)']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 학습 데이터와 검증 데이터 준비
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.3, random_state=511)

# Create a RandomForestRegressor model
model = RandomForestRegressor(random_state=42)  # No n_estimators specified

# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],  # You can adjust this list of values
    'max_depth': [None, 10, 20, 30],  # You can adjust this list of values
    # Add other hyperparameters here
}

# Create GridSearchCV object
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)

# Fit the model with GridSearchCV
grid_search.fit(X_train, y_train)

# Get the best estimator
best_rf_model = grid_search.best_estimator_

# Calculate RMSE and R2 for the best model
y_val_pred = best_rf_model.predict(X_val)
rmse_best = np.sqrt(mean_squared_error(y_val, y_val_pred))
r2_best = r2_score(y_val, y_val_pred)

print("Best hyperparameters:", grid_search.best_params_)

# Print RMSE and R2 for the best model
print("Validation RMSE (Best Model):", rmse_best)
print("Validation R2 Score (Best Model):", r2_best)

# Train the untuned model on the training data
model = RandomForestRegressor(random_state=42)  # No n_estimators specified
model.fit(X_train, y_train)

# Use the untuned model on the validation data
y_val_pred_no_tuning = model.predict(X_val)

# Calculate RMSE and R2 for the untuned model
rmse_no_tuning = np.sqrt(mean_squared_error(y_val, y_val_pred_no_tuning))
r2_no_tuning = r2_score(y_val, y_val_pred_no_tuning)

# Print RMSE and R2 for the untuned model
print("Validation RMSE (Untuned Model):", rmse_no_tuning)
print("Validation R2 Score (Untuned Model):", r2_no_tuning)

# Best hyperparameters: {'max_depth': 30, 'n_estimators': 300}
# Validation RMSE (Best Model): 18.662360034187646
# Validation R2 Score (Best Model): 0.8209394151223237
# Validation RMSE (Untuned Model): 18.685601406695557
# Validation R2 Score (Untuned Model): 0.8204931473295671