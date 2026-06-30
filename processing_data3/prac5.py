import statsmodels.api as sm
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 데이터 준비
data = pd.read_csv('../datasets3/08.03.01.customer_churn.csv')

X = data.drop(['churn', 'region_North', 'region_South', 'region_West'], axis = 1)
y = data['churn'] # 독립변수

# 상수항(constant) 추가 (필수!)
X = sm.add_constant(X)

# 모델 적합
model = sm.Logit(y, X).fit()

# 결과 확인
print(model.summary())

p_values = model.pvalues
print('유의하지 않은 독립변수 개수 : ', p_values[p_values > 0.05].count())
significant_vars = p_values[p_values <= 0.05]
print(significant_vars.head())

X_significant = data[['age', 'calls', 'monthly_charge']]
X_significant = sm.add_constant(X)

model_sig = sm.Logit(y, X_significant).fit()

#회귀계수의 평균
print('회귀계수의 평균 : ', model_sig.params.mean())

#calls 변수가 5 증가할 때 오즈비는 몇 배 증가하는지
call_coef = model.params['calls']
odds_ration = np.exp(call_coef*5)
print('calls 변수가 5 증가할 때 오즈비 : ', odds_ration)
