import statsmodels.api as sm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 데이터 준비
df = pd.read_csv('../datasets3/08.03.023.2.piq.csv')

# 종속변수
y = df['PIQ']

# 독립변수
X = df[['brain_size', 'height', 'weight', 
        'education_level', 'reading_habits', 'age']]

# train / test 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)

# 상수항 추가
X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

# OLS 모델 학습
model = sm.OLS(y_train, X_train)
result = model.fit()

# 결과 출력
print(result.summary())

# 각 독립변수의 회귀계수
print('회귀계수')
print(result.params)

# const 제외한 p-value
pvalues = result.pvalues.drop('const')

# 유의한 독립변수 개수
significant_vars = sum(pvalues < 0.05)
print('유의한 독립변수 개수 : ', significant_vars)

# 가장 유의미한 변수 = p-value가 가장 작은 변수
best_var = pvalues.idxmin()
best_coef = result.params[best_var]

print('가장 유의미한 변수 : ', best_var)
print('가장 유의미한 변수의 회귀계수 : ', best_coef)

# Test Data를 이용한 R^2
y_pred = result.predict(X_test)
r2 = r2_score(y_test, y_pred)

print('Test R^2 : ', r2)

# F통계량
print('f통계량 : ', result.fvalue)

# F통계량의 유의확률
print('f통계량의 유의수준 : ', result.f_pvalue)