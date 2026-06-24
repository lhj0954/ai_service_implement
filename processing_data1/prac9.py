import pandas as pd

df = pd.read_csv('../datasets/501_trash_bag.csv', encoding='cp949')
print(df.info())
print(df.head())