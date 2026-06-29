# 데이터 파일 읽기 예제
import pandas as pd
 
train = pd.read_csv('./datasets/07.02.01-sales_train_dataset.csv')
X_test = pd.read_csv('./datasets/07.02.02-sales_test_dataset_x.csv')

#데이터 확인하기
print(train.head())
print(train.info())
print(X_test.info())

#변수별 결측치 수 확인
#print(train.isnull().sum())
#print(X_test.isnull().sum())
 
#변수 구분
#print(train.columns) 

COL_DEL = ['BranchName']
COL_NUM = ['Population', 'IncomeGeneratingPopRatio', 'AverageIncome']
COL_CAT = ['City', 'IndustryType']
COL_Y = ['Sales']

#데이터 분할을 위해 X_train, y_train 데이터 별도 생성
X_train = train[COL_CAT + COL_NUM]
y_train = train[COL_Y]

#print(X_train.head())
#print('*'*100)
#print(y_train.head())

 #데이터 분할 훈련(7) vs 검증(3)
from sklearn.model_selection import train_test_split
X_tr, X_val, y_tr, y_val = train_test_split(X_train
                                            , y_train.values.ravel()
                                            , test_size=0.3)
#print(X_tr.head())
#print('*'*100)
#print(y_tr[:2])
#수치형 변수 - 데이터 스케일링
#print(X_tr.describe())

#표준화 수
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_tr[COL_NUM])

X_tr[COL_NUM]   = scaler.transform(X_tr[COL_NUM])
X_val[COL_NUM]  = scaler.transform(X_val[COL_NUM])
X_test[COL_NUM] = scaler.transform(X_test[COL_NUM])

print(X_tr[COL_NUM].head())

##범주형 변수 인코딩
from sklearn.preprocessing import LabelEncoder

X = pd.concat([X_train[COL_CAT], X_test[COL_CAT]])


for col in COL_CAT:
  le = LabelEncoder()
  le.fit(X[col])
  X_tr[col] = le.transform(X_tr[col])
  X_val[col] = le.transform(X_val[col])
  X_test[col] = le.transform(X_test[col])

  # 각 변수의 클래스 확인
  print(col)
  print(le.classes_)
  print('\n')

print(X_tr[COL_CAT].head())

#모형 학습
## 1) 랜덤 포레스트
from sklearn.ensemble import RandomForestRegressor
modelRF = RandomForestRegressor(random_state=123)
modelRF.fit(X_tr, y_tr)

## 2) XGBoost
from xgboost import XGBRegressor
modelXGB = XGBRegressor(objective='reg:squarederror', random_state=123)
modelXGB.fit(X_tr, y_tr)

#검증 데이터로 예측값 생성
y_val_predRF = modelRF.predict(X_val)
y_val_predXGB = modelXGB.predict(X_val)
    
## 답안 채점 기준인 rmse 사용 (mean_squared_error를 사용하여 검증해도 됩니다.)
from sklearn.metrics import mean_squared_error
import numpy as np

def cal_rmse(actual, pred):
  return np.sqrt(mean_squared_error(actual, pred))

#학습, 검증 성능 확인 함수,
def get_scores(model, X_tr, X_val, y_tr, y_val):

  y_tr_pred = model.predict(X_tr)
  y_val_pred = model.predict(X_val)
  tr_score = cal_rmse(y_tr, y_tr_pred)
  val_score = cal_rmse(y_val, y_val_pred)

  return f'train: {round(tr_score, 4)}, valid: {round(val_score, 4)}'


#모델별 성능 출력
print('Random Forest \t-', get_scores(modelRF, X_tr, X_val, y_tr, y_val))
print('XGBoost \t-', get_scores(modelXGB, X_tr, X_val, y_tr, y_val))

#하이퍼 파라미터 튜닝
modelXGB2 = XGBRegressor(objective='reg:squarederror', n_estimators=50, max_depth=5, min_child_weight=2, random_state=123)
modelXGB2.fit(X_tr, y_tr)

modelXGB3 = XGBRegressor(objective='reg:squarederror', n_estimators=50, max_depth=10, min_child_weight=2, random_state=123)
modelXGB3.fit(X_tr, y_tr)

modelXGB4 = XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=5, min_child_weight=1, random_state=123)
modelXGB4.fit(X_tr, y_tr)

modelXGB5 = XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=10, min_child_weight=2, random_state=123)
modelXGB5.fit(X_tr, y_tr)

print('XGBoost2 \t-', get_scores(modelXGB2, X_tr, X_val, y_tr, y_val))
print('XGBoost3 \t-', get_scores(modelXGB3, X_tr, X_val, y_tr, y_val))
print('XGBoost4 \t-', get_scores(modelXGB4, X_tr, X_val, y_tr, y_val))
print('XGBoost5 \t-', get_scores(modelXGB5, X_tr, X_val, y_tr, y_val))


#최종 결과값 생성
pred = modelXGB4.predict(X_test[COL_CAT+COL_NUM])
result = pd.DataFrame({'BranchName': X_test.BranchName, 'Sales': pred})
print(result.head())

## 최종 결과 확인 후 to_csv 함수로 제출
result.to_csv('003000000.csv', index=False)