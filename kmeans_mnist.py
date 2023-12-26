import torch
from torchvision import datasets, transforms
from sklearn.cluster import KMeans
import numpy as np
from collections import Counter

transform = transforms.Compose([transforms.ToTensor()])
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# 选择前N张图片和标签
N = 100
data = mnist_trainset.data[:N].float()
targets = mnist_trainset.targets[:N]

# 将28*28尺寸的图像数据转为784维的向量
data = data.view(data.size(0), -1)

# 使用KMeans进行聚类
K=30
kmeans = KMeans(n_clusters=K, n_init=K, max_iter=100, random_state=42)
kmeans.fit(data)

# 打印每个聚类中的标签
labels = kmeans.labels_

# 打印每个簇中图像的原始标签
for i in range(K):
    cluster_indices = torch.where(torch.tensor(labels) == i)[0]
    print(f"Cluster {i}: Labels: {targets[cluster_indices]}")

# 计算每个簇中最常见的标签
representative_labels = []
for i in range(K):
    cluster_indices = np.where(labels == i)[0]
    cluster_labels = targets[cluster_indices]
    most_common_label = Counter(cluster_labels.numpy()).most_common(1)[0][0]
    representative_labels.append(most_common_label)

# 计算准确率
correct_predictions = 0
total_predictions = len(labels)

for i in range(K):
    cluster_indices = np.where(labels == i)[0]
    cluster_labels = targets[cluster_indices]
    correct_predictions += np.sum(cluster_labels.numpy() == representative_labels[i])

accuracy = correct_predictions / total_predictions
print(f"K-Means Clustering Accuracy: {accuracy * 100:.2f}%")