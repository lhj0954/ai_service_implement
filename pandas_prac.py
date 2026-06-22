import pandas as pd

'''
data = {'Name': ['Alice', 'Bob', None],
'Age': [25, None, 30],
'City': ['New York', 'Los Angeles', None]}
df = pd.DataFrame(data)
'''

url = 'titanic.csv'
dataframe = pd.read_csv(url)
dataframe[dataframe['Age'].isnull()].head(2)

print(dataframe.tail(5)) # 각 값이 결측치인지 확인
