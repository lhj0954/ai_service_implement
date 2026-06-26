import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

train_df = pd.read_csv("../datasets2/06.02.01-Metaverse Training Data.csv")
test_x_df = pd.read_csv("../datasets2/06.02.02-Metaverse Testing Data_x.csv")
test_y_df = pd.read_csv("../datasets2/06.02.03-Metaverse Testing Data_y.csv")

#결측치 처리 생략 - info 확인 시 결측치가 없었다
COL_DEL = ['id']
COL_NUM = ['Age', 'Annual Salary']
COL_CAT = train_df.columns[:5].tolist()
COL_Y = ['Preferred Metaverse Type']

#train 데이터를 X, y 변수로 분할
X_train = train_df[COL_CAT + COL_NUM]
y_train = train_df[COL_Y] #2차원 구조의 DataFrame

#train data를 학습과 검증 데이터로 분할
#train_test_split(2D 데이터, 1D 배열 데이터, ...)
X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train.values.ravel(),
    test_size=0.3,
    stratify = y_train.values.ravel())

#범주 데이터 전처리 - label encoding
X = pd.concat([X_train[COL_CAT], test_x_df[COL_CAT]])
for col in COL_CAT :
    le = LabelEncoder()
    le.fit(X[col])
    X_train[col] = le.transform(X_train[col])
    X_val[col] = le.transform(X_val[col])
    test_x_df[col] = le.transform(test_x_df[col])

    #print(col)
    #print(le.classes_)

#스케일링
scaler = StandardScaler()
scaler.fit(X_train[COL_NUM])
X_train[COL_NUM] = scaler.transform(X_train[COL_NUM])
X_val[COL_NUM] = scaler.transform(X_val[COL_NUM])
test_x_df[COL_NUM] = scaler.transform(test_x_df[COL_NUM])

#모델학습
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

modelLR = LogisticRegression()
modelLR.fit(X_train, y_train)

modelRF = RandomForestClassifier()
modelRF.fit(X_train, y_train)

modelXGB = XGBClassifier()
modelXGB.fit(X_train, y_train)


y_val_predLR = modelLR.predict(X_val)
y_val_predRF = modelRF.predict(X_val)
y_val_predXGB = modelXGB.predict(X_val)

#성능지표
from sklearn.metrics import accuracy_score

f1scoreLR = f1_score(y_val, y_val_predLR, average='macro')
f1scoreRF = f1_score(y_val, y_val_predRF, average='macro')
f1scoreXGB = f1_score(y_val, y_val_predXGB, average='macro')

#print('LogisticRegression', f1scoreLR)
#print('RandomForestClassifier', f1scoreRF)
#print('XGBClassifier', f1scoreXGB)

#학습 / 검증 데이터 모델 성능 확인 함수 정의
def get_score(model, X_tr, X_val, y_tr, y_val) :
    y_tr_pred = model.predict(X_tr)
    y_val_pred = model.predict(X_val)
    tr_score = f1_score(y_tr, y_tr_pred, average='macro')
    val_score = f1_score(y_val, y_val_pred, average='macro')

    return f'train : {round(tr_score, 4)}, valid : {round(val_score, 4)}'

print('LogisticRegression', get_score(modelLR, X_train, X_val, y_train, y_val))
print('RandomForestClassifier', get_score(modelRF, X_train, X_val, y_train, y_val))
print('XGBClassifier', get_score(modelXGB, X_train, X_val, y_train, y_val))

#하이퍼 파라미터 튜닝
modelXGB2 = XGBClassifier(n_estimators = 50, max_depth = 3, min_child_weight = 1, random_state = 12)
modelXGB2.fit(X_train, y_train)

modelXGB3 = XGBClassifier(n_estimators = 50, max_depth = 3, min_child_weight = 2, random_state = 12)
modelXGB3.fit(X_train, y_train)

modelXGB4 = XGBClassifier(n_estimators = 50, max_depth = 5, min_child_weight = 1, random_state = 12)
modelXGB4.fit(X_train, y_train)

modelXGB5 = XGBClassifier(n_estimators = 100, max_depth = 5, min_child_weight = 1, random_state = 12)
modelXGB5.fit(X_train, y_train)

print('XGBoost2 \t-', get_score(modelXGB2, X_train, X_val, y_train, y_val))
print('XGBoost3 \t-', get_score(modelXGB3, X_train, X_val, y_train, y_val))
print('XGBoost4 \t-', get_score(modelXGB4, X_train, X_val, y_train, y_val))
print('XGBoost5 \t-', get_score(modelXGB5, X_train, X_val, y_train, y_val))

#참고
#print(help(f1_score))
