import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 날씨 데이터 통합
weather2017 = pd.read_csv('data/weather/weather2017.csv', encoding='cp949')
weather2018 = pd.read_csv('data/weather/weather2018.csv', encoding='cp949')
weather2019 = pd.read_csv('data/weather/weather2019.csv', encoding='cp949')
weather2020 = pd.read_csv('data/weather/weather2020.csv', encoding='cp949')
weather2021 = pd.read_csv('data/weather/weather2021.csv', encoding='cp949')
weather2022 = pd.read_csv('data/weather/weather2022.csv', encoding='cp949')

weather = pd.concat([weather2017, weather2018, weather2019, weather2020, weather2021, weather2022])
#print(weather.columns)

# 날씨 데이터 전처리
