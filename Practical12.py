import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd

# Load the iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

# 1. Create box plot for each numeric feature
plt.figure(figsize=(10, 8))
df.iloc[:, :4].boxplot()
plt.title("Box plots of iris dataset features")
plt.ylabel("Measurements (cm)")
plt.xticks(rotation=45)
plt.show()

# 2. Identifying outliers using IQR
for col in df.columns[:4]:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower) | (df[col] > upper)][col]

    print(f"\nFeature: {col}")
    print(f"Outliers:\n{outliers if not outliers.empty else 'No significant outliers'}")