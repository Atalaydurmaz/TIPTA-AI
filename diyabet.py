import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans 
from sklearn.decomposition import PCA # Görselleştirme için

# (1) Tıp Veri Kümesini Yükleme ve Ön İşleme
# ==========================================================
data = load_diabetes()
X = data.data        # 10 adet özellik (BMI, Kan Basıncı, Glukoz vb.)
feature_names = data.feature_names
n_samples = X.shape[0]

# Veri Standardizasyonu (Centroid mesafeye dayalı olduğu için zorunludur)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# (2) K-Means Modelini Oluşturma ve Eğitme
# ==========================================================
# Centroid modeli için K sayısını belirliyoruz
n_clusters = 3 

# KMeans modeli (K-Means, Centroid kümeleme modeline bir örnektir)
model = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')

# Model Eğitimi (Veri noktaları 3 merkeze atanır)
model.fit(X_scaled)
labels = model.labels_
cluster_centers = model.cluster_centers_ # Küme Merkezleri

# (3) Görselleştirme için Boyut İndirgeme (PCA)
# ==========================================================
# 10 boyutlu veriyi çizmek için, 2 boyuta indirelim.
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Küme merkezlerini de PCA uzayına yansıtma
centers_pca = pca.transform(cluster_centers)


# (4) Sonuçları Görselleştirme (PCA'nın İlk 2 Bileşeni Üzerinde)
# ==========================================================
plt.figure(figsize=(10, 6))

# Veri noktalarını çiz (rengi küme etiketine göre)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', 
            s=50, edgecolors='k', alpha=0.7)

# Küme merkezlerini (Centroids) çiz
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], marker='X', s=200, 
            c='red', label='Küme Merkezleri (Centroids)')

plt.title(f"K-Means (Centroid) Kümeleme: Diyabet Veri Seti ({n_clusters} Küme)")
plt.xlabel("Temel Bileşen 1")
plt.ylabel("Temel Bileşen 2")
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