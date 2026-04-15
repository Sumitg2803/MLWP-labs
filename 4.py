# Import required libraries 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 

# Step 1: Load dataset 
df = pd.read_csv("Iris.csv") 
 
# Step 2: Select features (exclude target column) 
X = df.iloc[:, :-1] 
 
# Step 3: Use Elbow Method to find optimal number of clusters 
wcss = []  # Within-Cluster Sum of Squares 
 
for i in range(1, 11): 
    kmeans = KMeans(n_clusters=i, random_state=42) 
    kmeans.fit(X) 
    wcss.append(kmeans.inertia_) 
 
# Step 4: Plot Elbow Graph 
plt.plot(range(1, 11), wcss) 
plt.xlabel("Number of Clusters") 
plt.ylabel("WCSS") 
plt.title("Elbow Method") 
plt.show() 
 
# Step 5: Train final model with optimal clusters (k=3) 
model = KMeans(n_clusters=3, random_state=42) 
model.fit(X) 
 
# Step 6: Get cluster labels 
labels = model.labels_ 
 
# Step 7: Visualize clusters (using first two features) 
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels) 
plt.xlabel("Feature 1") 
plt.ylabel("Feature 2") 
plt.title("Iris Clusters") 
plt.show() 