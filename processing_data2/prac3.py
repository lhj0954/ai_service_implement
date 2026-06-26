import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

train_df = pd.read_csv("../datasets2/2.0.subway_train.csv")
test_x_df = pd.read_csv("../datasets2/2.0.subway_test_x.csv")
test_y_df = pd.read_csv("../datasets2/2.0.subway_test_y.csv")

# 결측치 처리
visibility_mean = train_df['visibility'].mean()
station_name_mode = train_df['station_name'].mode()[0]

train_df['visibility'] = train_df['visibility'].fillna(visibility_mean)
train_df['station_name'] = train_df['station_name'].fillna(station_name_mode)

test_x_df['visibility'] = test_x_df['visibility'].fillna(visibility_mean)
test_x_df['station_name'] = test_x_df['station_name'].fillna(station_name_mode)

# date 처리
# date는 문자열이라 모델에 바로 못 넣음
# 날짜에서 day만 뽑아서 사용
train_df['date'] = pd.to_datetime(train_df['date'])
test_x_df['date'] = pd.to_datetime(test_x_df['date'])

train_df['day'] = train_df['date'].dt.day
test_x_df['day'] = test_x_df['date'].dt.day

train_df = train_df.drop(columns=['date'])
test_x_df = test_x_df.drop(columns=['date'])

# 독립변수 X, 종속변수 y 분리
X = train_df.drop(columns=['num_people'])
y = train_df['num_people']

''' date를 따로 이렇게 빼고 종속 독립 분리함
X = train_df.iloc[:, -1,1]
y = 
'''

# 학습 / 검증 데이터 분리
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=43
)

# 복사본 만들기
X_train = X_train.copy()
X_val = X_val.copy()
test_x_df = test_x_df.copy()

# 인코딩
cat_cols = ['day_of_week', 'station_name']

for col in cat_cols:
    le = LabelEncoder()
    
    # train, val, test에 있는 범주를 같이 보고 인코딩 규칙 생성
    le.fit(pd.concat([X_train[col], X_val[col], test_x_df[col]]))
    
    X_train[col] = le.transform(X_train[col])
    X_val[col] = le.transform(X_val[col])
    test_x_df[col] = le.transform(test_x_df[col])

# 스케일링 (거리, 크기를 비교하는 모델 KNN, SVR, 정규화 회귀, 신경망등은 스케일일이 필요)
#-> 스케일링 필요 없음

# 머신러닝 모델 생성 - 회귀 모델 사용
rf = RandomForestRegressor(random_state=1234)
rf.fit(X_train, y_train)

# 검증 데이터 예측
val_pred = rf.predict(X_val)

# RMSE 계산
rmse = np.sqrt(mean_squared_error(y_val, val_pred))
print(f'RMSE : {rmse:.2f}')

# 테스트 데이터 예측
test_pred = rf.predict(test_x_df)

# 제출 파일 생성
result = pd.DataFrame({
    'num_people': test_pred
})

result.to_csv("result.csv", index=False)

from sklearn.metrics import r2_score
modelL = LinearRegression()
modelL.fit(X_train, y_train)
y_pred_linear = modelL.predict(X_val)
r2_linear = r2_score(y_val, y_pred_linear)
rmse_linear = np.sqrt(mean_squared_error(y_val, y_val_pred))

modelRF = RandomForestRegressor()
modelRF.fit(X_train, y_train)
y_pred_RF = modelRF.predict(X_val)
r2_RF = r2_score(y_val, y_pred_RF)
rmse_linear = np.sqrt(mean_squared_error(y_val, y_pred_RF))

modelXGB = XGBRegressor()
modelXGB.fit(X_train, y_train)
y_pred_XGB = modelXGB.predict(X_val)
r2_XGB = r2_score(y_val, y_pred_XGB)
rmse_linear = np.sqrt(mean_squared_error(y_val, y_pred_XGB))