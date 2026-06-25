import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
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
X_train = train_df[COL_DEL + COL_NUM]
y_train = train_df[COL_Y] #2차원 구조의 DataFrame

#train data를 학습과 검증 데이터로 분할
#train_test_split(2D 데이터, 1D 배열 데이터, ...)
X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train.values.raval(),
    test_size=0.3,
    stratify = y_train.values.raval())

#범주 데이터 전처리 - label encoding
le = LabelEncoder()

for col in COL_CAT :
    train_df[col] = le.fit_transform(train_df[col])
    test_x_df[col] = le.fit_transform(test_x_df[col])

#스케일링
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
test_x_scaled = scaler.transform(COL_CAT)

#모델학습
rf = RandomForestClassifier(random_state=1234)
rf.fit(X_train, y_train)

val_pred = rf.predict(X_val)

모델 3개를 비교해서 하이퍼파라이터 비교
나머지 문제