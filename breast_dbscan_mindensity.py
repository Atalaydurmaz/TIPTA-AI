# -*- coding: utf-8 -*-
"""
Created on Sun Nov  2 10:39:12 2025

@author: Fatma
"""

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

# 3️⃣ Ölçekleme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4️⃣ DBSCAN Modeli (Minimum Gürültü İçin Optimize Edilmiş Parametreler)
# ==========================================================
# eps=2.5 ve min_samples=5, gürültü sayısını sıfıra yakınsatarak
# neredeyse tüm noktaları kümelere dahil etmelidir.
db = DBSCAN(eps=2.5, min_samples=5) 
labels = db.fit_predict(X_scaled)

# 5️⃣ Sonuç Özeti
# ==========================================================
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print("--- DBSCAN Sonuç Özeti ---")
print(f"Kullanılan Parametreler: eps={db.eps}, min_samples={db.min_samples}")
print(f"Elde Edilen Küme Sayısı: {n_clusters}")
print(f"Gürültü noktası sayısı (Noise): {n_noise}")

# 6️⃣ Görselleştirme (2D PCA ile)
# ==========================================================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 7))
# Gürültü noktalarını gri (veya siyah) renkte, kümeleri renkli çizelim
plt.scatter(X_pca[labels == -1, 0], X_pca[labels == -1, 1], c='gray', s=20, label='Gürültü (-1)', alpha=0.6)
plt.scatter(X_pca[labels != -1, 0], X_pca[labels != -1, 1], c=labels[labels != -1], cmap='viridis', s=50, label='Kümeler')

plt.title(f"DBSCAN: Minimum Gürültü ile Kümeleme (eps=2.5, min_samples=5)")
plt.xlabel(f"PCA1 ({pca.explained_variance_ratio_[0]*100:.1f}% Varyans)")
plt.ylabel(f"PCA2 ({pca.explained_variance_ratio_[1]*100:.1f}% Varyans)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()