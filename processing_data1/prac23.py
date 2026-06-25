import pandas as pd
import numpy as np

df = pd.read_csv("../datasets/09.03.employees.csv")

# 1) training_score 결측값: 해당 부서의 중앙값으로 대체
df["training_score"] = df["training_score"].fillna(
    df.groupby("department")["training_score"].transform("median")
)

# 2) years_of_service 결측값: 성과등급별 근속연수 평균값으로 대체
# 평균값은 소수점 이하 올림
service_mean = df.groupby("performance_rating")["years_of_service"].transform(
    lambda x: np.ceil(x.mean())
)

df["years_of_service"] = df["years_of_service"].fillna(service_mean)

# (1) salary / years_of_service 기준 상위 3번째 직원의 근속연수
df["salary_per_service"] = df["salary"] / df["years_of_service"]

third_emp = df.sort_values("salary_per_service", ascending=False).iloc[2]

print("상위 3번째 직원:", third_emp["emp_id"])
print("근속연수:", int(third_emp["years_of_service"]))

# (2) salary / training_score 기준 상위 5번째 직원의 교육점수와 부서
df["salary_per_training"] = df["salary"] / df["training_score"]

fifth_emp = df.sort_values("salary_per_training", ascending=False).iloc[4]

print("상위 5번째 직원:", fifth_emp["emp_id"])
print("교육점수:", int(fifth_emp["training_score"]))
print("부서:", fifth_emp["department"])