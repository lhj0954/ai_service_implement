import pandas as pd

df = pd.read_csv('../datasets/0603-Crime Data.csv')

dg = df.groupby('Police Station')
print(dg.head())