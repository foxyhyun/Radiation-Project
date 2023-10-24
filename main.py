import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, LSTM, Dropout


# === 데이터 불러오기 및 통합 === 
weather2017 = pd.read_csv('data/weather/weather2017.csv', encoding='cp949')
weather2018 = pd.read_csv('data/weather/weather2018.csv', encoding='cp949')
weather2019 = pd.read_csv('data/weather/weather2019.csv', encoding='cp949')
weather2020 = pd.read_csv('data/weather/weather2020.csv', encoding='cp949')
weather2021 = pd.read_csv('data/weather/weather2021.csv', encoding='cp949')
weather2022 = pd.read_csv('data/weather/weather2022.csv', encoding='cp949')


#print(weather.shape) ==> (52563, 38)
nsv = pd.read_csv('data/nsv/jeonnam.csv', encoding='cp949')
# print(nsv.shape) ==> (191003, 17)

# === 날씨 데이터 전처리 ===
weather = pd.concat([weather2017, weather2018, weather2019, weather2020, weather2021, weather2022])
# 1. NaN 개수 확인하기
# print(weather.isnull().sum()) # ==> 지울 것들 : 기온 QC플래그, 강수량 (mm), 강수량 QC플래그, 풍속 QC플래그, 풍향 QC플래그, 습도 QC플래그, 현지기압 QC플래그, 해면기압 QC플래그, 일조 (hr), 일조 QC플래그, 일사(MJ/m2), 일사 QC플래그, 적설 (cm), 3시간신적설 (cm), 운형(운형약어), 최저운고 (100m ), 지면상태(지면상태코드), 현상번호(국내식), 지면온도 QC플래그
weather = weather.drop(['기온 QC플래그','강수량(mm)','전운량(10분위)','중하층운량(10분위)','강수량 QC플래그','풍속 QC플래그','풍향 QC플래그','습도 QC플래그','현지기압 QC플래그','해면기압 QC플래그','일조(hr)','일조 QC플래그','일사(MJ/m2)','일사 QC플래그','적설(cm)','3시간신적설(cm)','운형(운형약어)','최저운고(100m )','지면상태(지면상태코드)','현상번호(국내식)','지면온도 QC플래그'], axis=1)
#print(nsv.isnull().sum())
#print(weather.isnull().sum())

# 2. 결측값 채우기(NaN => 이전값 대치)
weather = weather.fillna(method='ffill')
#print(weather.isnull().sum())

# 3. 불필요한 변수 제거
weather = weather.drop(['지점','지점명'],axis=1)
nsv = nsv.drop(['id','device_id','data_type','CDMA_tel','InteTemp','ElecTemp','ip_addr','DoseRate_uR','DoseRate_low','DoseRate_high','InteTemp_high','ElecTemp_high','state','DoseRate_warn','DoseRate_alert'],axis=1)

# 4. 날짜 포맷 맞추기
weather['time'] = pd.to_datetime(weather['일시'])
nsv['time'] = pd.to_datetime(nsv['rcv_time'])
weather = weather.drop(['일시'], axis=1)
nsv = nsv.drop(['rcv_time'], axis=1)
# 5. 데이터 통합
df = pd.merge(weather, nsv, on='time', how='inner')

# 6. 기타
df = df.rename(columns={'DoseRate_nSv': 'nsv'})
order = ['time','기온(°C)', '풍속(m/s)', '풍향(16방위)', '습도(%)', '증기압(hPa)', '이슬점온도(°C)',
       '현지기압(hPa)', '해면기압(hPa)', '시정(10m)', '지면온도(°C)', '5cm 지중온도(°C)',
       '10cm 지중온도(°C)', '20cm 지중온도(°C)', '30cm 지중온도(°C)','nsv']
df = df[order]

# df.to_csv('data//mergeData.csv')

# === 상관계수 파악 ===
# 다중공선성 문제를 일으키는 변수 제거
# 꼭 필요한 변수는 성능이 안좋더라도 사용하기로 결정
df = df.drop(['해면기압(hPa)','5cm 지중온도(°C)','10cm 지중온도(°C)', '20cm 지중온도(°C)', '30cm 지중온도(°C)'],axis=1)

# === Random Forest로 중요도 확인 후 상위 5개만 사용 ===
df = df.drop(['증기압(hPa)','풍향(16방위)','이슬점온도(°C)','습도(%)'], axis=1)

# === MinMax === 

# MinMaxScaler 객체 생성
scaler = MinMaxScaler()

# 스케일링을 적용할 데이터를 선택
data_to_scale = df[['기온(°C)', '풍속(m/s)', '현지기압(hPa)', '시정(10m)', '지면온도(°C)', 'nsv']]

# MinMax 변환
scaled_data = scaler.fit_transform(data_to_scale)
scaled_df = pd.DataFrame(scaled_data, columns=data_to_scale.columns)

# 'time' 변수와 스케일링된 데이터를 합침
df = pd.concat([df['time'], scaled_df], axis=1)

train_year = [2017, 2018, 2019, 2020] 
val_year = 2021  
test_year = 2022

df['year'] = df['time'].dt.year

df_train = df[df['year'].isin(train_year)]
df_val = df[df['year'] == val_year]
df_test = df[df['year'] == test_year]

X_train = df_train.drop(['time','nsv','year'], axis=1)
X_val = df_val.drop(['time','nsv','year'], axis=1)
X_test = df_test.drop(['time','nsv','year'], axis=1)
y_train = df_train['nsv']
y_val = df_val['nsv']
y_test = df_test['nsv']

# === ANN Model===
# model = Sequential()
# model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(1, activation='linear'))

# model.compile(loss='mean_squared_error', optimizer='adam')

# model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

# y_pred = model.predict(X_test)

# mse = mean_squared_error(y_test, y_pred)
# mae = mean_absolute_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'Mean Absolute Error (MAE): {mae:.2f}')
# print(f'R-squared (R2) 값: {r2:.2f}')

# R2 = 0.15

# === CNN Model === 

# model = Sequential()

# # 1D Convolutional Layer 추가
# model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))

# # Max Pooling Layer 추가
# model.add(MaxPooling1D(pool_size=2))

# # Flatten Layer 추가
# model.add(Flatten())

# # 완전 연결 레이어 추가
# model.add(Dense(64, activation='relu'))
# model.add(Dense(1, activation='linear'))  # 선형 활성화 함수 (회귀 문제)

# # 모델 컴파일
# model.compile(optimizer='adam', loss='mean_squared_error')  # MSE를 손실 함수로 사용 (회귀 문제)

# # 데이터 준비 (CNN은 3D 데이터 형태여야 하므로 차원 추가)
# X_train_cnn = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
# X_val_cnn = X_val.values.reshape(X_val.shape[0], X_val.shape[1], 1)

# # 모델 학습
# model.fit(X_train_cnn, y_train, validation_data=(X_val_cnn, y_val), epochs=50, batch_size=32)

# # 테스트 데이터로 예측
# X_test_cnn = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)
# y_pred = model.predict(X_test_cnn)

# # 평가 지표 계산
# mse = mean_squared_error(y_test, y_pred)
# mae = mean_absolute_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f"Mean Squared Error (MSE): {mse:.2f}")
# print(f"Mean Absolute Error (MAE): {mae:.2f}")
# print(f"R-squared (R2) 값: {r2:.2f}")
# R2 = 0.14

# === LSTM Model === 
# model = Sequential()
# model.add(LSTM(50, activation='tanh', input_shape=(X_train.shape[1], 1)))
# model.add(Dropout(0.2))
# model.add(Dense(1))

# # 모델 컴파일
# model.compile(optimizer='adam', loss='mean_squared_error')

# # 데이터 준비 (LSTM은 3D 데이터 형태여야 하므로 차원 추가)
# X_train_lstm = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
# X_val_lstm = X_val.values.reshape(X_val.shape[0], X_val.shape[1], 1)

# # 모델 학습
# model.fit(X_train_lstm, y_train, validation_data=(X_val_lstm, y_val), epochs=10, batch_size=32)

# # 테스트 데이터로 예측
# X_test_lstm = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)
# y_pred = model.predict(X_test_lstm)

# # 평가 지표 계산
# mse = mean_squared_error(y_test, y_pred)
# mae = mean_absolute_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f"Mean Squared Error (MSE): {mse:.2f}")
# print(f"Mean Absolute Error (MAE): {mae:.2f}")
# print(f"R-squared (R2) 값: {r2:.2f}")





