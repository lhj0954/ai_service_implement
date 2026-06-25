import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

train_df = pd.read_csv("../datasets2/09.02.train.csv")
test_x_df = pd.read_csv("../datasets2/09.02.test_x.csv")
test_y_df = pd.read_csv("../datasets2/09.02.test_y.csv")

#결측치 처리 생략 - info 확인 시 결측치가 없었다

#범주 데이터 전처리 - label encoding
le = LabelEncoder()
cat_cols = ['education_level', 'loan_type']

for col in cat_cols :
    train_df[col] = le.fit_transform(train_df[col])
    test_x_df[col] = le.fit_transform(test_x_df[col])

feature_cols = train_df.columns.drop(['customer_id', 'credit_rating']).tolist()
#feature_cols = train_df.columns[-1:1].tolist()
X_train, X_val, y_train, y_val = train_test_split(
    train_df[feature_cols],
    train_df['credit_rating'],
    test_size=0.3,
    random_state=1234)

#스케일링
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
test_x_scaled = scaler.transform(test_x_df[feature_cols])

#모델학습
rf = RandomForestClassifier(random_state=1234)
rf.fit(X_train, y_train)

val_pred = rf.predict(X_val)

#f1 - 정밀도와 재현율의 조화평균
val_f1 = f1_score(y_val, val_pred, average = 'macro')#다중 분류 분석에서 평균계산 방식
print(f'Macro로 평균 계산한 f1 score : {val_f1:.2f}')

#특성 중요도 확인
importance = pd.DataFrame({'feature': feature_cols, 'importance': rf.feature_importances_})
importances = importance.sort_values('importance', ascending = False)
print(importances)

selected_features = importances['feature'].head(8).tolist()

#선택된 특성으로 분류 분석 모델 학습
X_train_selected = X_train[:, [feature_cols.index(feat) for feat in selected_features]]
X_val_selected = X_val[:, [feature_cols.index(feat) for feat in selected_features]]
test_x_selected = test_x_scaled[:, [feature_cols.index(feat) for feat in selected_features]]

rf_final = RandomForestClassifier(random_state=1234)
rf_final.fit(X_train_selected, y_train)

#8개의 특성 학습 모델 성능 확인
val_pred_final = rf_final.predict(X_val_selected)
val_f1_final = f1_score(y_val, val_pred_final, average='macro')
print(f'8개 특성 학습 후 Macro로 평균 계산한 f1 score : {val_f1_final:.2f}')

y_pred = rf_final.predict(test_x_selected)
result = pd.DataFrame({'index' : range(len(test_x_selected)), 'y_pred' : y_pred})

result.to_csv('2020039060.csv', index = False)