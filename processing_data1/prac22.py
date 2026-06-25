import pandas as pd

df = pd.read_csv("../datasets/09.02.crime.csv")

# 검거율 계산
df["arrest_rate"] = df["arrests"] / df["incidents"] * 100

# 가장 마지막 분기
last_quarter = df["quarter"].max()

# 마지막 분기 데이터만 추출
last_df = df[df["quarter"] == last_quarter]

# 마지막 분기에서 검거율이 가장 높은 범죄유형
target_crime = last_df.sort_values("arrest_rate", ascending=False).iloc[0]["crime_type"]

# 해당 범죄유형의 검거건수 총합
answer = df[df["crime_type"] == target_crime]["arrests"].sum()

print("마지막 분기:", last_quarter)
print("검거율이 가장 높은 범죄유형:", target_crime)
print("검거건수 총합:", answer)