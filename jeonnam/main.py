import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
from function import evaluate_model
from model import xgboost_model

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

train_years = [2017, 2018, 2019, 2020]
val_year = 2021
df['year'] = df['time'].dt.year
# Filter data for training and validation
train_mask = df['year'].isin(train_years)  # Train on data from 2017 to 2020
val_mask = df['year'] == val_year          # Validate on data from 2021

X_train = df[train_mask].drop(['time', 'nsv', 'year'], axis=1)
y_train = df[train_mask]['nsv']

X_val = df[val_mask].drop(['time', 'nsv', 'year'], axis=1)
y_val = df[val_mask]['nsv']
window_size = 20

# Test Data (Last week of 2022)
test_year = 2022
test_mask = (df['year'] == test_year) & (df['time'] >= '2022-12-01')  # Select data from the last week of 2022

X_test = df[test_mask].drop(['time', 'nsv', 'year'], axis=1)
y_test = df[test_mask]['nsv']

# === XGBoost === 
xgboost_model(X_train,y_train, X_val, y_val)


