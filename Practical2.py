import pandas as pd

#----------------------------------------------
# 1. Load the Dataset
#----------------------------------------------
df = pd.read_csv("venv\Datasets\Telecom Churn.csv")

print("\n=== First 5 rows of Dataset ===")
print(df.head())

print("\n=== Columns in dataset ===")
print(df.columns)

# ----------------------------------------------
# 2. Mininmum value of each feature
# ----------------------------------------------
print("\n=== Mininmum value of each feature ===")
print(df.min())

#----------------------------------------------
# 3. Maximum value of each feature
#----------------------------------------------
print("\n=== Maximum value of each feature ===")
print(df.max())

#----------------------------------------------
# 4. Mean value of each feature
#----------------------------------------------
print("\n=== Mean of each column ===")
print(df.mean(numeric_only=True))

#----------------------------------------------
# 5. Std deviation of each feature
#----------------------------------------------
print("\n=== Std deviation of each column ===")
print(df.std(numeric_only=True))

#----------------------------------------------
# 6. Variance of each feature
#----------------------------------------------
print("\n=== Variance of each column ===")
print(df.var(numeric_only=True))

#----------------------------------------------
# 7. Percentiles (25th, 50th, 75th)
#----------------------------------------------
print("\n=== 25th Percentile ===")
print(df.select_dtypes(include="number").quantile(0.25))

print("\n=== 50th Percentile (Median) ===")
print(df.select_dtypes(include="number").quantile(0.50))

print("\n=== 75th Percentile ===")
print(df.select_dtypes(include="number").quantile(0.75))