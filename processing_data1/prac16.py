import pandas as pd

df = pd.read_csv('../datasets/0702-fish_weight_data.csv')
print(df.head())

df1 = df.copy().drop