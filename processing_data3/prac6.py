import statsmodels.api as sm
import pandas as pd
import numpy as np

# 데이터 준비
df = pd.read_csv('../datasets3/07.03.01-gender_prediction_dataset.csv')
df1 = df.copy()
df1['Gender'] = df1['Gender'].map({'Male' : 1, 'Female' : 0})

X = df1.iloc[:, :-1]
y = df1['Gender']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size = 0.3,
    stratify = y
)
# 상수항(constant) 추가 (필수!)
X_train = sm.add_constant(X_train)

# 모델 적합
model = sm.Logit(y_train, X_train).fit()

# 결과 확인
print(model.summary())

#Weight의 오즈비
weight_coef = model.params['Weight']
odds_ration = np.exp(weight_coef)
print('Weight 오즈비 : ', odds_ration)

#로짓우도
print('모델의 로짓우도 : ', model.llf)

#예측한 결과와 실제값의 오차율
X_test = sm.add_constant(X_test)
y_pred = (model.predict(X_test) >= 0.5).astype(int)
print(round(np.mean(abs(y_pred.values - y_test)), 4))