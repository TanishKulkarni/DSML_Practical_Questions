import os
import random 
import numpy as np
import pandas as pd

k = 3
ITERATIONS = 10
RANDOM_SEED = 42
csv_paths_to_try = ["IRIS.csv", "iris.csv", r'venv\Datasets\IRIS.csv', r'venv/Datasets/IRIS.csv']

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# LOAD THE DATASET
df = None
for p in csv_paths_to_try:
    if os.path.exists(p):
        df = pd.read_csv(p)
        print(f"Loaded dataset from: {p}")
        break

if df is None:
    # fallback
    try:
        import seaborn as sns
        df = sns.load_dataset("iris")
        print("Loaded inbuilt seaborn 'iris' dataset (fallback)")
    except Exception as e:
        raise FileNotFoundError("Could not find IRIS.csv in tried paths and seaborn load failed")
    
# PREPARE FEATURE MATRIX
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_cols) < 1:
    raise ValueError("No numeric columns found in dataset. Check your csv")

# Typical iris features are the first 4 numeric columns
feature_cols = numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols
X = df[feature_cols].copy().to_numpy(dtype=float)

print("\nUsing feature columns for clustering:", feature_cols)
print("Data shape:", X.shape)

# K-MEANS IMPLEMENTATION
n_samples, n_features = X.shape

# Initialize centroids by randomly choosing k distinct points from X
initial_idx = np.random.choice(n_samples, size=k, replace=False)
centroids = X[initial_idx].astype(float)
print("\nInitial centroids (chosen from random data points)")
for i, c in enumerate(centroids):
    print(f"Centroid {i}: {c}")

# iterate assign -> update
for it in range(1, ITERATIONS + 1):
    distances = np.linalg.norm(X[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)

    # Assign labels: nearest centroid
    labels = np.argmin(distances, axis=1)

    # Update centroid
    new_centroids = np.zeros_like(centroids)
    for k in range(k):
        members = X[labels == k]
        if len(members) == 0:
            reinit_idx = np.random.choice(n_samples, 1)[0]
            new_centroids[k] = X[reinit_idx]
            print(f"Interation {it}: Cluster {k} had no members - reinitialized to data point index")
        else:
            new_centroids[k] = members.mean(axis=0)

    centroids = new_centroids

# FINAL RESULTS
print("\n Final cluster means after", ITERATIONS, "iterations:")
for i, c in enumerate(centroids):
    formatted = ", ".join(f"{val:.4f}" for val in c)
    print(f"Cluster {i} mean: [{formatted}]")


    
