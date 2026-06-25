import pandas as pd

df = pd.read_csv("../datasets/09.01.bank_loan.csv")

# 1) 연령층별 P은행, S은행 대출 평균
mean_df = df.groupby("age_group")[["p_bank_loan", "s_bank_loan"]].mean()

print(mean_df)

# 2) 연령층별 두 은행 대출금액 차이
mean_df["loan_diff"] = abs(mean_df["p_bank_loan"] - mean_df["s_bank_loan"])

max_age_group = mean_df["loan_diff"].idxmax()

print(mean_df["loan_diff"])
print("차이가 가장 큰 연령층:", max_age_group)

# 3) 차이가 가장 큰 연령층에서 개인별 차이가 가장 작은 고객번호
df["customer_diff"] = abs(df["p_bank_loan"] - df["s_bank_loan"])

target = df[df["age_group"] == max_age_group]

answer_customer = target.sort_values("customer_diff").iloc[0]["customer_id"]

print("정답 고객번호:", answer_customer)