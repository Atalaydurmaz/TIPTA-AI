# 1️⃣ Kütüphaneler
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 2️⃣ Veri Kümesini Yükle
data = load_breast_cancer()
X = data.data
y = data.target  # 0 = malignant, 1 = benign

# 3️⃣ Ölçekleme (DBSCAN mesafeye dayalı çalıştığı için çok önemli)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4️⃣ DBSCAN Modeli
db = DBSCAN(eps=1.7, min_samples=8)
labels = db.fit_predict(X_scaled)

# 5️⃣ Sonuç Özeti
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f"Küme sayısı: {n_clusters}")
print(f"Gürültü noktası sayısı: {n_noise}")

# 6️⃣ Görselleştirme (2D PCA ile)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=40)
plt.title("DBSCAN - Breast Cancer Veri Kümesi Kümeleme Sonucu")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.colorbar(label="Küme Etiketleri")
plt.show()
