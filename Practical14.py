import pandas as pd

# Load dataset
df = pd.read_csv(r'venv\Datasets\Covid Vaccine Statewise.csv')

# Show available columns (important)
print("Columns in dataset:\n")
print(df.columns)

# Display first rows
print("\nDataset Preview:\n")
print(df.head())

# Show structure / data types
print("\nDataset Info:\n")
print(df.info())

# Summary statistics
print("\nStatistical Summary:\n")
print(df.describe())

# ---- FIX COLUMN NAMES HERE ----
# Check the exact column names from printed df.columns
# Common correct names:
# "Male(Individuals Vaccinated)"
# "Female(Individuals Vaccinated)"

# Replace these with the exact names from your dataset
male_col = "Male(Individuals Vaccinated)"
female_col = "Female(Individuals Vaccinated)"

# 2. Number of Males Vaccinated
male_vaccinated = df.groupby("State")[male_col].max().sort_values(ascending=False)
print("\nState-wise Males Vaccinated:\n")
print(male_vaccinated)

# 3. Number of Females Vaccinated
female_vaccinated = df.groupby("State")[female_col].max().sort_values(ascending=False)
print("\nState-wise Females Vaccinated:\n")
print(female_vaccinated)
