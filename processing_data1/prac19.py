import pandas as pd

df = pd.read_csv("../datasets/Auto_sales.csv")

# SUV 판매 비율 계산
df["suv_ratio"] = df["suv_sales"] / df["total_sales"]

# SUV 판매 비율이 세 번째로 높은 나라
third_suv_ratio_country = df.sort_values("suv_ratio", ascending=False).iloc[2]
sedan_value = third_suv_ratio_country["sedan_sales"]

# 세단 판매량이 네 번째로 높은 나라
fourth_sedan_country = df.sort_values("sedan_sales", ascending=False).iloc[3]
suv_value = fourth_sedan_country["suv_sales"]

answer = sedan_value + suv_value

print(answer)