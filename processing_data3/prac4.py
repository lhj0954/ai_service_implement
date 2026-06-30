import statsmodels.api as sm
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 데이터 준비
data = pd.read_csv('../datasets3/09.03.02.insurance_claim.csv')

y = data['claim'] # 0 또는 1
X = data[['age', 'bmi', 'bp', 'glucose']] # 독립변수

# 상수항(constant) 추가 (필수!)
X = sm.add_constant(X)

# 모델 적합
model = sm.Logit(y, X)
result = model.fit()

# 결과 확인
print(result.summary())
chi2_stats = result.llr
chi2_pvalue = result.llr_pvalue
print('카이제곱 통계량 : ',chi2_stats)
print('카이제곱 유의확률 : ',chi2_pvalue)

significant_vars = sum(result.pvalues < 0.05)
print('유의한 독립변수 개수 : ', significant_vars)

#혈압과 혈당이 증가할 때의 오즈비
Odds_Ratio = np.exp(result.params)
print('혈압 오즈비 : ', Odds_Ratio['bp'])
print('혈당 오즈비 : ', Odds_Ratio['glucose'])