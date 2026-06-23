import pandas as pd

df = pd.read_csv('../datasets/303_titanic.csv')

miss_cnt = df.isnull().sum()
max_col = miss_cnt.idxmax()
max_val = miss_cnt.max()

print(f"{max_col} : {max_val}")