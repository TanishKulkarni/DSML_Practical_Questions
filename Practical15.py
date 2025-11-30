import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv('venv\Datasets\Titanic.csv')

# Show first few rows
print("Dataset Preview:\n")
print(df.head())

# Basic Info
print("\nDataset Info:")
print(df.info())

# 1. SURVIVAL COUNT PLOT
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Survived", palette="viridis")
plt.title("Survival Count")
plt.show()

# 2. SURVIVALS BY GENDER
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Sex", hue="Survived", palette="viridis")
plt.title("Survival by Gender")
plt.show()

# 3. SURVIVALS BY PASSENGER CLASS
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Pclass", hue="Survived", palette="viridis")
plt.title("Survival by Passenger Class")
plt.show()

# 4. AGE DISTRIBUTION
plt.figure(figsize=(7,5))
sns.histplot(data=df, x="Age", bins=50, kde=True)
plt.title("Age Distribution of Passengers")
plt.show()