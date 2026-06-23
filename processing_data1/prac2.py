import pandas as pd

df = pd.read_csv('../datasets/202_housing.csv')

#print(df.info())
n = int(len(df) * 0.8)
df_80 = df.loc[n-1, 'total_bedrooms']

df_80_std = df_80.std()
median = df_80.median()
med_df_80 = df_80.fillna(median)
med_df_80_std = med_df_80.std()

print(abs(df_80_std - med_df_80_std))