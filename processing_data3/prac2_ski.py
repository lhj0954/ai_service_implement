import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('../datasets3/08.03.023.2.piq.csv')

y = df['PIQ']
features = ['brain_size', 'height', 'weight',
            'education_level', 'reading_habits', 'age']
X = df[features]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

print('절편 : ', model.intercept_)

coef_df = pd.DataFrame({
    '변수': features,
    '회귀계수': model.coef_
})
print(coef_df)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

print('Test R^2 : ', r2)

# 표준화 계수 기준 가장 영향력 큰 변수
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

scaled_model = LinearRegression()
scaled_model.fit(X_train_scaled, y_train)

standard_coef_df = pd.DataFrame({
    '변수': features,
    '표준화_회귀계수': scaled_model.coef_,
    '절댓값': np.abs(scaled_model.coef_)
}).sort_values(by='절댓값', ascending=False)

print(standard_coef_df)

print('가장 영향력이 큰 변수 : ', standard_coef_df.iloc[0]['변수'])

# 새 참가자 예측
new_person = pd.DataFrame({
    'brain_size': [90],
    'height': [175],
    'weight': [70],
    'education_level': [2],
    'reading_habits': [1],
    'age': [25]
})

pred_piq = model.predict(new_person)
print('예측 PIQ : ', pred_piq[0])