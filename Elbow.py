# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 22:02:41 2025

@author: Fatma
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# (1) Veri Yükleme ve Ön İşleme
# ==========================================================
data = load_diabetes()
X = data.data
# K-Means mesafeye dayalı olduğu için standardizasyon zorunludur.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# (2) Elbow Metodu Uygulaması
# ==========================================================

# Farklı K değerleri için WCSS (Inertia) değerlerini tutacak liste
wcss = [] 

# Deney yapılacak K değerleri aralığı
k_range = range(1, 11) # K=1'den K=10'a kadar deneyeceğiz.

for k in k_range:
    # KMeans modelini oluştur
    # inertia_ k-means'in WCSS değerini hesaplar
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    
    # Modeli eğit
    kmeans.fit(X_scaled)
    
    # WCSS (Inertia) değerini listeye ekle
    wcss.append(kmeans.inertia_)

# (3) Sonuçların Görselleştirilmesi (Elbow Grafiği)
# ==========================================================
plt.figure(figsize=(9, 5))
plt.plot(k_range, wcss, marker='o', linestyle='--', color='blue')
plt.title('Elbow (Dirsek) Metodu ile Optimal K Değerini Belirleme')
plt.xlabel('Küme Sayısı (K)')
plt.ylabel('WCSS (Küme İçi Kareler Toplamı)')
plt.grid(True, alpha=0.5)

# Grafikteki "Dirsek" noktasını manuel olarak işaretleyebiliriz (Tahmin)
# Diyabet verisi için genellikle optimal K=3 veya K=4 civarındadır.
# 4 noktasına bir ok ekleyelim
optimal_k = 4
plt.annotate(f'Dirsek Noktası (Tahmini K={optimal_k})', 
             xy=(optimal_k, wcss[optimal_k-1]), 
             xytext=(optimal_k + 1, wcss[optimal_k-1] + 100),
             arrowprops=dict(facecolor='red', shrink=0.05))

plt.show()

# (4) Sonuç Çıkarımı
# ==========================================================
print("\n--- Elbow Metodu Sonucu ---")
print("WCSS değerleri K=1'den K=10'a kadar listelenmiştir:")
for k, inertia in zip(k_range, wcss):
    print(f"K={k}: WCSS = {inertia:.2f}")

print(f"\nGrafiğe bakıldığında, WCSS'teki azalmanın keskinliğini kaybettiği 'dirsek' noktası genellikle K={optimal_k} civarındadır. Bu, en uygun küme sayısının {optimal_k} olabileceğini gösterir.")