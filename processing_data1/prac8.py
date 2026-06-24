import pandas as pd

df = pd.read_csv('../datasets/403_netflix.csv')

jan_2018 = df['date_added'].str.contains('January')&df['release_year']

print(jan_2018)