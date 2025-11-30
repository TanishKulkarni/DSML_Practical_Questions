import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("venv\Datasets\Lung Cancer.csv")

print("\n==== ORIGINAL DATA (HEAD) =====")
print(df.head())

print("\n==== DATA INFO BEFORE CLEANING ====")
print(df.info())

# 1. DATA CLEANING
# 1.1 Remove duplicate rowss
df = df.drop_duplicates()

# 1.2 Standardize column names (remove spaces, lowercases)
df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()

print("\n==== COLUMN NAMES CLEANED ====")
print(df.columns)

# 1.3 Detect missing values
print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

# 1.4 Handle missing values
# Option A: Fill numeric with mean
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

# Option B: Fill categorical with mode
cat_cols = df.select_dtypes(include=['object']).columns
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

print("\n===== AFTER HANDLING MISSING VALUES =====")
print(df.isnull().sum())

# 1.5 Fix inconsistent text values (capitalize)
for col in cat_cols:
    df[col] = df[col].astype(str).str.strip().str.capitalize()

# 1.6 Convert categorical numbers stored as strings into numeric
for col in df.columns:
    if df[col].dtype == 'object':
        if df[col].str.isnumeric().all():
            df[col] = pd.to_numeric(df[col])

# ------------------------------------------
# 2. DATA TRANSFORMATION
# ------------------------------------------

# 2.1 Label Encoding for categorical features
df_encoded = df.copy()
for col in df_encoded.select_dtypes(include=['object']).columns:
    df_encoded[col] = df_encoded[col].astype('category').cat.codes

print("\n===== LABEL-ENCODED DATA (HEAD) =====")
print(df_encoded.head())

# 2.2 Min-Max Normalization for numeric columns
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df_scaled = df_encoded.copy()
df_scaled[num_cols] = scaler.fit_transform(df_scaled[num_cols])

print("\n===== NORMALIZED DATA (HEAD) =====")
print(df_scaled.head())

# 2.3 Create derived features (example)
if 'age' in df.columns:
    df['age_group'] = pd.cut(df['age'], bins=[0, 40, 60, 100], labels=['Young', 'Middle', 'Old'])

print("\n===== DATA WITH NEW FEATURE (HEAD) =====")
print(df.head())

# 2.4 Rename columns for clarity
df = df.rename(columns={'gender':'sex'}) if 'gender' in df.columns else df

# 2.5 Drop irrelevant columns (example)
cols_to_drop = ['id', 'patient_id']  
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

print("\n===== FINAL CLEANED & TRANSFORMED DATA (HEAD) =====")
print(df.head())

# Save the cleaned dataset
df.to_csv("Lung_Cancer_Cleaned.csv", index=False)
print("\nCleaned dataset saved as Lung_Cancer_Cleaned.csv")