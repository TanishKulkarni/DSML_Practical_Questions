import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd

# Load the iris dataset
iris = load_iris()

# Create a dataframe for easier handling
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

# 1. List down features and their types
print("Feature names and data types:")
print(df.dtypes)

# 2. Create historgram for each numeric feature
df.hist(figsize=(10, 8), color='skyblue', edgecolor='black')
plt.suptitle("Histogram of Iris Dataset Features", fontsize=16)
plt.show()