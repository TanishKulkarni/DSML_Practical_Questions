# DESCRIBE THE DATASET
import pandas as pd

# Load dataset
df = pd.read_csv('venv\Datasets\Covid Vaccine Statewise.csv')

# Show first few rows
print("Dataset Preview:\n")
print(df.head())

# Show structure
print("\n Dataset Info:")
print(df.info())

# Summary statistics
print("\nStatistical Summary:")
print(df.describe())

# 2. NUMBER OF PERSONS STATE_WISE VACCINATED FOR FIRST DOSE
first_dose = df.groupby("State")["First Dose Administered"].max().sort_values(ascending=False)
print("\nState-wise First Dose Vaccination:\n")
print(first_dose)

# 3. NUMBER OF PERSONS STATE_WISE VACCINATED FOR SECOND DOSE
second_dose = df.groupby("State")["Second Dose Administered"].max().sort_values(ascending=False)
print("\nState-wise Second dose vaccination:\n")
print(second_dose)