import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 데이터 불러오기
data = pd.read_csv('data/df.csv', encoding='cp949')
data.dropna(inplace=True)

# 필요한 변수만 추출
X = data[['기온(°C)', '누적강수량(mm)', '현지기압(hPa)', '해면기압(hPa)', '일조(Sec)']]
y = data['선량률(nSv/h)']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.3, random_state=511)

# Create a Linear Regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Use the model on the validation data
y_val_pred = model.predict(X_val)

# RMSE and R2 scores
rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
r2 = r2_score(y_val, y_val_pred)
print("Validation RMSE:", rmse)
print("Validation R2 Score:", r2)

# Validation RMSE: 43.76398674668988
# Validation R2 Score: 0.015307506444740882