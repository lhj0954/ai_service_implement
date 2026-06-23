import pandas as pd

df = pd.read_csv('../datasets/301_housing.csv')
#print(df.info())
df = df.dropna().reset_index(drop=True)

n = int(len(df) * 0.7)
df = df.iloc[:n]

Q1 = df['housing_median_age'].quantile(0.25)

print(Q1)