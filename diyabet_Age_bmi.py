# -*- coding: utf-8 -*-
"""
Created on Sun Nov  2 10:20:48 2025

@author: Fatma
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# (1) Tıp Veri Kümesini Yükleme ve Ön İşleme
# ==========================================================
data = load_diabetes()
X = data.data       # 10 adet özellik
feature_names = data.feature_names

# Veri Standardizasyonu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# (2) K-Means Modelini Oluşturma ve Eğitme
# ==========================================================
n_clusters = 3
model = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
model.fit(X_scaled)

labels = model.labels_
cluster_centers = model.cluster_centers_

# (3) Görselleştirme İçin Özellik İndekslerini Belirleme
# ==========================================================
# Orijinal özellik isimlerinden 'age' ve 'bmi' indekslerini bulalım.
age_index = feature_names.index('age')
bmi_index = feature_names.index('bmi')

# Görselleştirme için ölçeklenmiş veriden sadece bu iki sütunu seçelim.
X_plot = X_scaled[:, [age_index, bmi_index]]
centers_plot = cluster_centers[:, [age_index, bmi_index]]

# (4) Sonuçları Görselleştirme (Age vs. BMI Üzerinde)
# ==========================================================
plt.figure(figsize=(10, 6))

# Veri noktalarını çiz (rengi küme etiketine göre)
# X ekseni: 'age', Y ekseni: 'bmi'
plt.scatter(X_plot[:, 0], X_plot[:, 1], c=labels, cmap='viridis', 
            s=50, edgecolors='k', alpha=0.7)

# Küme merkezlerini (Centroids) çiz
plt.scatter(centers_plot[:, 0], centers_plot[:, 1], marker='X', s=200, 
            c='red', label='Küme Merkezleri (Centroids)')

plt.title(f"K-Means Kümeleme: Yaş ve BMI İlişkisi ({n_clusters} Küme)")
plt.xlabel("Yaş (Ölçeklenmiş Değer)")
plt.ylabel("BMI (Ölçeklenmiş Değer)")
plt.legend()
plt.grid(True, alpha=0.6)
plt.show()

# (5) Tıbbi Yorumlama: Centroid Değerlerini İnceleme
# ==========================================================
print("\n--- Tıbbi Yorumlama (Centroid Değerleri) ---")

# Küme merkezlerini orijinal özellik isimleriyle DataFrame'e dönüştürme
df_centers = pd.DataFrame(cluster_centers, columns=feature_names)
df_centers.index.name = 'Küme'

# Centroid değerlerini inceleyelim (Ölçeklenmiş değerlerdir)
print("3 Kümenin Centroid (Ortalama Özellik) Değerleri:")
print(df_centers.round(2))