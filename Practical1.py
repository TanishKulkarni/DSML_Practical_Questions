import pandas as pd

# 1. Reading data from different formats: CSV and Excel
df_csv = pd.read_csv('E:\\DSML Practical\\venv\\Datasets\\Titanic.csv')

# Print first 5 rows of csv
print("\n=== First 5 rows of CSV ===")
print(df_csv.head())

# Indexing and Selecting

#Selecting a single column
print("\n=== Selecting a single column: Name ===")
print(df_csv["Name"].head())

# Select multiple columns
print("\n=== Select multiple columns ===")
print(df_csv[["PassengerId", "Name", "Age", "Survived"]].head())

# Select a row using the iolc (Position-based)
print("\n=== Select 5th row using iloc ===")
print(df_csv.iloc[4])

# Boolean indexing: passengers older than 50
print("\n=== Passengers older than 50 ===")
print(df_csv[df_csv["Age"] > 50].head())

# 3. Sorting the data 

# Sort the data by age(ascending)
print("\n=== Sorting by Age (ascending) ===")
print(df_csv.sort_values(by="Age").head())

# Sort by class then age (multi-sort)
print("\n=== Sorting by Pclass then Age ===")
print(df_csv.sort_values(by=["Pclass", "Age"]).head())

# 4. Describe Attributes
print("\n=== Describe numeric attributes ===")
print(df_csv.describe())

print("\n=== Describe all attributes ===")
print(df_csv.describe(include="all"))

# 5. Check data types of each column
print("\n=== Data types (dtypes) ===")
print(df_csv.dtypes)
