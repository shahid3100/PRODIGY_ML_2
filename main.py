import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load dataset
data = pd.read_csv("Mall_Customers.csv")

# Select features (Best for K-Means)
X = data[['AnnualIncome', 'SpendingScore']]

# Apply K-Means
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X)

# Add cluster labels to dataset
data['Cluster'] = clusters

# Visualize clusters
plt.figure(figsize=(8,6))
plt.scatter(X['AnnualIncome'], X['SpendingScore'], 
            c=clusters, cmap='viridis')

plt.scatter(kmeans.cluster_centers_[:,0], 
            kmeans.cluster_centers_[:,1], 
            s=300, c='red', label='Centroids')

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segmentation using K-Means")
plt.legend()
plt.show()

# Print clustered data
print(data)
