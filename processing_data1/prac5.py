import pandas as pd

df = pd.read_csv('../datasets/302_worlddata.csv')
#print(df.info())
df_2000 = df[df['year']==2000].drop('year', axis = 1)
print(df_2000.head())
df_2000.index = ['Value']
print(df_2000.head())
dfT = df_2000.T
print(len(dfT[dfT['Value'] > dfT['Value'].mean()]))