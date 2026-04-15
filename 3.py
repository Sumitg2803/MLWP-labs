# Import required libraries 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
 
# Step 1: Load dataset 
df = pd.read_csv("Mall_Customers.csv") 
 
# Step 2: Select features for clustering 
X = df[['Annual Income (k$)', 'Spending Score (1-100)']] 
 
# Step 3: Create and train KMeans model 
model = KMeans(n_clusters=5, random_state=42) 
model.fit(X) 
 
# Step 4: Get cluster labels 
labels = model.labels_ 
 
# Step 5: Visualize clusters 
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels) 
plt.xlabel("Annual Income (k$)") 
plt.ylabel("Spending Score (1-100)") 
plt.title("Customer Segments") 
plt.show()