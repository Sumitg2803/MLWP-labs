import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score


df = pd.read_csv("Mall_Customers.csv")

X = df.drop("CustomerID",  axis=1)
X = df.drop("Gender", axis=1)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

plt.figure(figsize=(8,6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap="viridis")

plt.title("K-Means Clusters")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.show()

S = silhouette_score(X_scaled, labels)
DB = davies_bouldin_score(X_scaled, labels)

print("Silhouette Score:", S)
print("Davies-Bouldin Index:", DB)