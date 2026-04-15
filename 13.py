import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

df = pd.read_csv("diabetes.csv")

X = df[["Glucose", "BMI"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

# Plot clusters
plt.figure(figsize=(8,6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap="viridis")

plt.xlabel("Glucose")
plt.ylabel("BMI")
plt.title("K-Means Clustering")
plt.colorbar(label="Cluster")
plt.show()

S= silhouette_score(X_scaled, labels)
DB= davies_bouldin_score(X_scaled, labels)

print(f"Silhouette Score:", S)
print(f"Davies-Bouldin Index:", DB)