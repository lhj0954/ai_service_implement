import statsmodels.api as sm
import pandas as pd
import numpy as np

# 데이터 준비
df = pd.read_csv('../datasets3/09.03.01.house_price.csv')
print(df.info())

y = df['price'] # 종속변수 (연속형)
X = df[['floor', 'built_year', 'subway_distance']] # 독립변수
'''
X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]
'''

# OLS 모델 적합
model = sm.OLS(y, X)
result = model.fit()
# 결과 출력
print(result.summary())
print('f통계량 : ', result.fvalue) #f통계량
print('f통계량의 유의수준 : ', result.f_pvalue) #f통계량의 유의확률

significant_vars = sum(result.pvalues[1:]<0.05)
print('유의한 독립변수 개수 : ', significant_vars)

features = ['area', 'floor', 'built_year', 'subway_distance']
best_r2 = 0
best_combo = None

from itertools import combinations
for combo in combinations(features, 2) :
    X_subset = df[list(combo)]
    # 상수항(constant) 추가 → 절편(β0)을 위해
    X_subset = sm.add_constant(X_subset)
    model_subset = sm.OLS(y, X_subset).fit()
    if model_subset.rsquared > best_r2 :
        best_r2 = model_subset.rsquared
        best_combo = combo

print('best combo : ', best_combo)

