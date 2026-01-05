# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 15:06:07 2025

@author: Fatma
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage 

# (1) Tıp Veri Kümesini Yükleme ve Ön İşleme
# ==========================================================
# Wisconsin Meme Kanseri Veri Kümesi (Tanı: Benign/Malign)
data = load_breast_cancer()
X = data.data        # Hücre çekirdeği özellikleri (30 özellik)
feature_names = data.feature_names
target_names = data.target_names

print(f"Veri kümesi boyutu: {X.shape}")

# Veri Standardizasyonu (Önemli! Tıp/biyolojik verilerde ölçek farkı büyük olabilir)
# Özelliklerin ortalamasını 0 ve standart sapmasını 1 yapar.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Hedef küme sayısını 2 olarak belirleyelim 
# (Veride 2 tanı kategorisi olduğu için bu sayıyı seçtik, ancak kümeleme etiketleri bilmez)
n_clusters = 2 

# (2) Hiyerarşik Kümeleme Modelini Oluşturma ve Eğitme
# ==========================================================
# linkage='ward' metodu ile kümeleme yapıyoruz
model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward') #modelin nasıl kümeleme yapacağını belirtiyor.

# Kümeleme işlemini gerçekleştir
model.fit(X_scaled)

# Elde edilen küme etiketleri (0 veya 1)
labels = model.labels_

# (3) Dendrogram Çizimi (Hiyerarşiyi Gösterme)
# ==========================================================
# Dendrogram, kümeleme hiyerarşisinin nasıl oluştuğunu gösterir.
Z = linkage(X_scaled, method='ward')
#veriler üzerinde hiyerarşik birleştirme adımlarını hesaplayıp, dendrogram fonksiyonuna uygun bir biçimde sonuç döndürür.
plt.figure(figsize=(15, 7))
dendrogram(
    Z,
    truncate_mode='lastp',  # Son p birleşmeyi göster
    p=30,                   # Son 30 birleşmeyi göster (daha okunur olması için)
    leaf_rotation=90.,  ## Yaprak etiketlerini 90° döndür (okunabilirlik için)
    leaf_font_size=8.,   # Etiket yazı boyutu
    show_contracted=True, # Kırpılmış birleşmeleri özetle göster
)
plt.title('Meme Kanseri Veri Seti Hiyerarşik Kümeleme Dendrogramı')
plt.xlabel(f'Örnek İndeksi veya Birleşen Küme Sayısı (Toplam {len(X)})')
plt.ylabel('Mesafe')
plt.show()

# (4) Sonuçları Görselleştirme (2 Boyutta)
# ==========================================================
# Veri 30 boyutlu olduğu için, en önemli iki özelliği seçerek saçılım grafiği çizelim.
# "mean radius" ve "mean texture" genellikle önemlidir.
idx_radius = np.where(feature_names == 'mean radius')[0][0]
idx_texture = np.where(feature_names == 'mean texture')[0][0]

plt.figure(figsize=(10, 6))
plt.scatter(X_scaled[:, idx_radius], X_scaled[:, idx_texture], c=labels, cmap='coolwarm', 
            s=50, edgecolors='k')

plt.title(f"Hiyerarşik Kümeleme: '{feature_names[idx_radius]}' vs '{feature_names[idx_texture]}'")
plt.xlabel(feature_names[idx_radius] + " (Ölçeklenmiş)")
plt.ylabel(feature_names[idx_texture] + " (Ölçeklenmiş)")
plt.colorbar(label='Küme Etiketi', ticks=range(n_clusters))
plt.grid(True, alpha=0.6)
plt.show()