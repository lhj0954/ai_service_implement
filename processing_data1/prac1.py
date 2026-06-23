import pandas as pd

df = pd.read_csv('../datasets/201_boston.csv')
df_sorted = df.sort_values('CRIM', ascending = False)
tenth_crim = df_sorted.iloc[9, 0]
df.iloc[:10, 0] = tenth_crim
df_age80 = df_sorted[df_sorted['AGE']>=80]

print(df_age80.head(10))
print(df_age80['CRIM'].mean())