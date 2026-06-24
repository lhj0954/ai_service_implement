import pandas as pd

df = pd.read_csv('../datasets/07.03-apartment_prices_dataset.csv')
print(df.head())
print(df.info())

df1 = df.copy().drop('Price', axis = 1)
print(df1.info())

df1 = df1.dropna()
print(df1.head())

