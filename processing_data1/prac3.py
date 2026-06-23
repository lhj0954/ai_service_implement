import pandas as pd

df = pd.read_csv('../datasets/203_housing.csv')

#print(df.info())
pop_mean = df['population'].mean()
pop_std = df['population'].std()

lower = pop_mean - 1.5*pop_std
upper = pop_mean + 1.5*pop_std

Q1 = df['population'].quantile(0.25)
Q3 = df['population'].quantile(0.75)

IQR = Q3 - Q1
lower_ = Q1 - 1.5*IQR
uppper_ = Q3 + 1.5*IQR

outliers = df[(df['population'] > upper) | (df['population'] < lower)]
result = outliers['population'].sum()

print(result)

'''
이상치

평균 + 1.5*표준편차
평균 - 1.5*표준편차

q1 - 1.5*IQR
q3 + 1.5*IQR

=>quantile()
'''