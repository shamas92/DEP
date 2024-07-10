import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Loading the dataset
file_path = 'commodity_prices.csv'
data = pd.read_csv(file_path)

# Data cleaning
data_filled = data.fillna(data.mean())
data_preprocessed = data_filled.drop(columns='Year')

# Exploratory Data Analysis (EDA)
data_plot = data_preprocessed.copy()
data_plot['Year'] = data['Year']

plt.figure(figsize=(14, 8))
sns.lineplot(x='Year', y='Crude Oil', data=data_plot, label='Crude Oil')
sns.lineplot(x='Year', y='Gold', data=data_plot, label='Gold')
sns.lineplot(x='Year', y='Cocoa', data=data_plot, label='Cocoa')
plt.title('Price Trends of Select Commodities Over Years')
plt.ylabel('Price')
plt.xlabel('Year')
plt.legend()
plt.grid(True)
plt.show()

# Normalize the data
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data_preprocessed)

# Determining the optimal number of clusters using the Elbow method
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_normalized)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), sse, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.grid(True)
plt.show()

# Applying K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(data_normalized)

# Reduce dimensions for visualization using PCA
pca = PCA(n_components=2) 
reduced_data = pca.fit_transform(data_normalized)

plt.figure(figsize=(10, 8))
sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=clusters, palette='viridis', style=clusters, s=100)
plt.title('Visualization of Commodity Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()