import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset 
titanic = sns.load_dataset("titanic")

# Display first rows 
print(titanic.head())

# Plot histogram of ticket fare
plt.figure(figsize=(8,5))
sns.histplot(data=titanic, x="fare", bins=40, kde=True, color="blue")
plt.title("Distribution of Ticket Fare on the Titanic ")
plt.xlabel("Fare")
plt.ylabel("Frequency")
plt.show()