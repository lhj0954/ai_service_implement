import pandas as pd

df = pd.read_csv("../datasets/Co_Nmch.csv")

# 최대-최소 정규화
df_norm = (df - df.min()) / (df.max() - df.min())

# 각 컬럼의 표준편차
std_co = df_norm["Co"].std()
std_nmch = df_norm["Nmch"].std()

# 표준편차 차이
diff = abs(std_co - std_nmch)

print(std_co)
print(std_nmch)
print(diff)

# 변동성이 큰 변수
if std_co > std_nmch:
    print("Co")
else:
    print("Nmch")