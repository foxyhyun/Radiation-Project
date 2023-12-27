import streamlit as st
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
# Define merged_df as a global variable
merged_df = None

def set_korean_font():
    font_path = "NotoSansKR-VariableFont_wght.ttf"  # 다운로드한 폰트 파일의 경로로 수정

    # 폰트 매니저에 폰트 추가
    prop = font_manager.FontProperties(fname=font_path)

    # 기존 캐시를 비워줍니다.
    font_manager._fmcache = None

    # Matplotlib에 한글 폰트 설정
    plt.rc('font', family=prop.get_name())
    plt.rc('axes', unicode_minus=False)

def check_columns(file, prefix):
    global merged_df  # Use the global variable

    if file is not None:
        df = pd.read_csv(file)

        required_columns = ['date', 'nSv']
        if all(col in df.columns for col in required_columns):
            # 'date' 컬럼을 datetime으로 변환(문자열로 받아올때의 경우 대비)
            df['date'] = pd.to_datetime(df['date'])

            # 'nSv' 컬럼을 변경
            df[f'nSv_{prefix}'] = df['nSv']

            # 불필요한 열 제거
            df = df[['date', f'nSv_{prefix}']]

            merged_df = df  # Assign the modified DataFrame to the global variable
            return True, merged_df
        else:
            return False, None
    else:
        return False, None

st.title('선량 상관관계 예측 프로그램')

# 데이터 입력
st.subheader('데이터 입력')

# 주석 메시지
comment_message = "CSV 파일에는 'date'와 'nSv' 컬럼이 있어야합니다."
st.info(comment_message, icon="ℹ️")

file1 = st.file_uploader('nsv1 (csv or excel) : ', type=['csv', 'xls'])
file2 = st.file_uploader('nsv2 (csv or excel) : ', type=['csv', 'xls'])
file3 = st.file_uploader('nsv3 (csv or excel) : ', type=['csv', 'xls'])

# 파일이 선택되었는지 확인
if file1 is not None and file2 is not None and file3 is not None:
    # 'date'와 'nSv' 컬럼 확인
    columns_exist, df1 = check_columns(file1, "1")
    _, df2 = check_columns(file2, "2")
    _, df3 = check_columns(file3, "3")

    if columns_exist:
        # 결과 확인 버튼
        if st.button("데이터 분석하기"):
            progress_text = "Operation in progress. Please wait."
            my_bar = st.progress(0, text=progress_text)

            for percent_complete in range(100):
                time.sleep(0.01)
                my_bar.progress(percent_complete + 1, text=progress_text)

            # 각 데이터프레임을 합치는 과정을 진행합니다.
            merged_df = pd.merge(df1, df2, on='date', how='inner')
            merged_df = pd.merge(merged_df, df3, on='date', how='inner')

            # 합친 데이터프레임 생성
            time.sleep(1)  # 가상의 지연

            # 결과를 출력합니다.
            st.subheader('상위 5개 데이터 확인')
            st.dataframe(merged_df.head(5))  # 상위 5개의 행을 출력
            
            st.subheader('데이터 범위:')
            st.write(f"시작 날짜: {merged_df['date'].min()}, 끝 날짜: {merged_df['date'].max()}")


            st.subheader('상관관계 시각화')
            last_index = merged_df.columns[0]
            corr_values = merged_df.corr()['nSv_1'].drop(['date'])
            st.write(corr_values)

            # 히트맵 그래프
            st.subheader('상관관계 히트맵')
    
            # 한글 폰트 설정 적용
            set_korean_font()
    
            # Matplotlib figure 생성
            fig, ax = plt.subplots(figsize=(10, 6))
    
    # Seaborn 히트맵 그래프 그리기
    # x 및 y 축 레이블에 대한 폰트 설정 및 date 열 제외
            sns.heatmap(merged_df.drop(columns=['date']).corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax,
                xticklabels=merged_df.drop(columns=['date']).columns, yticklabels=merged_df.drop(columns=['date']).columns)
    
    # Streamlit에 플로팅
            st.pyplot(fig)


            # 돌아가기 버튼을 누르면 결과 확인이 종료됩니다.
            if st.button("돌아가기"):
                st.success("결과 확인이 종료되었습니다.")
                
            # 진행 상태 표시 제거
            my_bar.empty()

# ...




st.markdown(
    """<style>
        .stButton>button {
            width: 100%;
            text-align: center;
        }
    </style>""",
    unsafe_allow_html=True,
)



                




