import pandas as pd
import matplotlib.pyplot as plt

# 1. LOAD THE DATASET
df = pd.read_csv("venv\Datasets\House Data.csv")

print("\n=== first 5 rows of dataset ===")
print(df.head())

# Select numeric features only
numeric_df = df.select_dtypes(include="number")

print("\n=== Numeric features ===")
print(numeric_df.columns)

# 2. STANDARD DEVIATION
print("\n=== Standard Deviation ===")
print(numeric_df.std())

# 3. VARIANCE 
print("\n=== Variance ===")
print(numeric_df.var())

# 4. PERCENTILES (25TH, 50TH, 75TH)
print("\n=== Percentile ===")
print(numeric_df.quantile(0.25))

print("\n=== Median (50th percentile) ===")
print(numeric_df.quantile(0.50))

print("\n=== 75th Percentile ===")
print(numeric_df.quantile(0.75))

#. HISTOGRAM FOR EACH FEATURE
print("\n=== Create Histogram for each numeric feature===")

for column in numeric_df.columns:
    plt.figure(figsize=(6, 4))
    plt.hist(numeric_df[column].dropna(), bins=30)
    plt.title(f"Histogram of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.grid(False)

    # Save each histogram as an image
    plt.savefig(f"hist_{column}.png")

    plt.close()

print("Histograms saved as image files.")
print("=== DONE ===")