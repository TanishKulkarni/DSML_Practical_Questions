import pandas as pd

# Load your Dataset
df = pd.read_csv('venv\Datasets\House Data.csv')

# Show columns to identify categorical and numeric variables
print("Dataset Columns:\n")
print(df.columns)

# Check data types of each columns
print("\n Data Types of each column: ")
print(df.dtypes)

# Choose categorical and quantitative variable
categorical_var = "Type"
quantitative_var = "NumberFloorsofBuilding"

# Summary
summary = df.groupby(categorical_var)[quantitative_var].agg(
    ['mean', 'median', 'min', 'max', 'std']
)

print(f"\nSummary statistics of '{quantitative_var}' grouped by '{categorical_var}':\n")
print(summary)