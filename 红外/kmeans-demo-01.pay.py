import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 生成样本数据
X, y = make_blobs(n_samples=300, centers=4, random_state=42, cluster_std=0.60)

# 创建 KMeans 模型
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)

# 获取簇中心和簇标签
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# 可视化聚类结果
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200)
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
