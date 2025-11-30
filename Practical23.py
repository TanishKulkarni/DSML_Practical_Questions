import pandas as pd
import numpy as np
from math import log2

# Create the dataset manually exactly as shown in the table
data = {
    "Age": ["Young","young","Middle","Old","Old","Old","Middle","Young","Young","Old","Young",
            "Middle","Middle","Middle","Old"],
    "Income": ["High","High","High","Medium","Low","Low","Low","Medium","Low","Medium","Medium",
               "Medium","High","Medium","Medium"],
    "Married": ["No","No","No","No","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","No"],
    "Health": ["Fair","Good","Fair","Fair","Fair","Good","Good","Fair","Fair","Fair","Good","Good",
               "Fair","Good","Good"],
    "Class": ["No","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No"]
}

df = pd.DataFrame(data)

# Normalize Age values
df["Age"] = df["Age"].str.title()

print("\nDataset:\n")
print(df)

def entropy(series):
    values, counts = np.unique(series, return_counts=True)
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log2(probabilities))

# 1. Frequency table for Age vs Class
freq_table = pd.crosstab(df["Age"], df["Class"])
print("\nFrequency Table (Age vs Class):\n")
print(freq_table)

# 2. Parent entropy (entropy of Class)
parent_entropy = entropy(df["Class"])
print("\nParent Entropy H(Class) =", parent_entropy)

# 3. Compute weighted entropy for each Age group
age_groups = df["Age"].unique()

weighted_entropy = 0

print("\n--- Entropy for Each Age Group ---")
for age in age_groups:
    subset = df[df["Age"] == age]
    group_entropy = entropy(subset["Class"])
    weight = len(subset) / len(df)
    weighted_entropy += weight * group_entropy
    print(f"Age = {age}, Count = {len(subset)}, Entropy = {group_entropy:.4f}, Weight = {weight:.4f}")

# 4. Information Gain
information_gain = parent_entropy - weighted_entropy

print("\nWeighted Entropy after splitting on Age =", weighted_entropy)
print("\nInformation Gain IG(Age) =", information_gain)
