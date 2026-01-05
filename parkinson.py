# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 21:18:20 2025

@author: Fatma
"""

# 📦 Gerekli Kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs

def run_dbscan_pipeline_with_spread(X, eps=1.5, min_samples=5, title="DBSCAN Sonuçları"):
    """
    DBSCAN + PCA + yayılım görselleştirme
    Gürültü noktaları kırmızı çarpı ile gösterilir.
    """
    # Ölçekleme
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X_scaled)
    
    # PCA ile 2 boyuta indirgeme
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Görselleştirme
    plt.figure(figsize=(12,5))
    
    # Tüm noktaları döngü ile çiz
    unique_labels = set(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for k, col in zip(unique_labels, colors):
        class_member_mask = (labels == k)
        xy = X_pca[class_member_mask]
        if k == -1:
            plt.scatter(xy[:,0], xy[:,1], c='red', marker='x', s=50, label='Gürültü')
        else:
            plt.scatter(xy[:,0], xy[:,1], c=[col], s=40, label=f'Cluster {k}')
    
    plt.title(title)
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    plt.legend()
    plt.show()
    
    # Cluster bilgisi
    print(title)
    print(pd.Series(labels).value_counts())
    print("\n")
    return labels

# ==========================================================
# 1️⃣ Parkinson’s Disease Dataset
# ==========================================================
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
df_parkinson = pd.read_csv(url)
X_parkinson = df_parkinson.drop(['status', 'name'], axis=1)

labels_parkinson = run_dbscan_pipeline_with_spread(
    X_parkinson, eps=1.5, min_samples=5, title="DBSCAN - Parkinson’s Disease"
)

