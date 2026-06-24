import pandas as pd

df = pd.read_csv("../datasets/coffee_consumption.csv")

# 1) 대륙별 Arabica 평균이 가장 큰 대륙
continent_ara_most = df.groupby('continent')['Arabica_consumption'].mean().idxmax()

# 2) 그 대륙 데이터만 추출 후 Arabica 소비량 내림차순 정렬
df_continent = df[df['continent'] == continent_ara_most] \
                .sort_values('Arabica_consumption', ascending=False) \
                .reset_index(drop=True)

# 3) 다섯 번째 나라의 Arabica 소비량
answer = int(df_continent.iloc[4]['Arabica_consumption'])

print(continent_ara_most)
print(df_continent.iloc[4]['country'])
print(answer)