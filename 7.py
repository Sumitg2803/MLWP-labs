import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("Titanic.csv")

df = df.select_dtypes(include='number')
df.fillna(df.mean(), inplace=True)

# Compute correlation matrix
corr_matrix = df.corr()

# Correlation with target (Survived)
target_corr = corr_matrix["Survived"].sort_values(ascending=False)

print("Correlation with Target:\n")
print(target_corr)

# Visualization
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()