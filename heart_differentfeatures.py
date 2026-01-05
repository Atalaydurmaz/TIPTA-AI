# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 16:56:48 2025

@author: Fatma
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# (1) Sentetik Veri Oluşturma ve Hazırlama
# ==========================================================
# NOT: Sentetik veri, gerçek kalp hastalığı verisi ile aynı kesin ayrımı sağlamayabilir,
# ancak kümeleme mantığını göstermek için kullanılır.
np.random.seed(42)
data_size = 303
X = np.zeros((data_size, 5))

# Kalp Hastalığı Veri Setine Benzer Özellikler
X[:, 0] = np.random.normal(54, 8, data_size)    # Yaş (age)
X[:, 1] = np.random.randint(1, 5, data_size)    # Göğüs Ağrısı Tipi (chest_pain_type)
X[:, 2] = np.random.normal(245, 50, data_size) # Kolesterol (cholesterol) - Yüksek değerler
X[:, 3] = np.random.normal(150, 20, data_size) # Maks. Kalp Atış Hızı (max_heart_rate)
X[:, 4] = np.random.uniform(0, 4, data_size)    # oldpeak (Egzersize bağlı ST depresyonu) - Yüksek değerler

feature_names = ['age', 'chest_pain_type', 'cholesterol', 'max_heart_rate', 'oldpeak']

# Veri Standardizasyonu (Ölçekleme)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# (2) Hiyerarşik Kümeleme Modelini Oluşturma ve Eğitme
# ==========================================================
# Önceki grafikte belirlenen 4 küme sayısını kullanıyoruz
n_clusters = 4

model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')

# Kümeleme işlemini gerçekleştir (Model, 5 özelliğin hepsini kullanır)
model.fit(X_scaled)
labels = model.labels_

# (3) Dendrogram Çizimi - (Önceki görseli temsil eder)
# ==========================================================
Z = linkage(X_scaled, method='ward')
KESME_MESAFESI = 13.5

plt.figure(figsize=(15, 7))
dendrogram(
    Z,
    truncate_mode='lastp',
    p=30,
    leaf_rotation=90.,
    leaf_font_size=8.,
    show_contracted=True,
    color_threshold=KESME_MESAFESI
)

plt.axhline(y=KESME_MESAFESI, c='black', linestyle='--', linewidth=2, label=f'{n_clusters} Küme Eşiği ({KESME_MESAFESI} Mesafesi)')
plt.legend()

plt.title('Kalp Hastalığı Veri Seti Hiyerarşik Kümeleme Dendrogramı (Eşik Çizgisiyle)')
plt.xlabel(f'Hasta İndeksi veya Birleşen Küme Sayısı (Toplam {len(X)})')
plt.ylabel('Mesafe')
plt.show()

# (4) Sonuçları Yeni Özelliklerle Görselleştirme (2 Boyutta)
# ==========================================================
# Yeni iki önemli özelliği seçelim: 'cholesterol' ve 'oldpeak'
idx_cholesterol = feature_names.index('cholesterol')
idx_oldpeak = feature_names.index('oldpeak')

plt.figure(figsize=(10, 6))
# X ekseni: cholesterol, Y ekseni: oldpeak
# Renkler: 5 özellikten elde edilen küme etiketleri (labels)
plt.scatter(X_scaled[:, idx_cholesterol], X_scaled[:, idx_oldpeak], c=labels, cmap='Spectral',
            s=50, edgecolors='k')

# Bilgilendirme metnini grafiğe ekleme
# X ve Y Eksenleri: Grafiğin yerleşimi (noktaların konumu) sadece 2 özelliğe (cholesterol ve oldpeak) göre belirlenir.
# Renkler (c=labels): Noktaların rengi ise, modelin 5 özellikten elde ettiği 4 farklı küme etiketine göre atanır.

plt.title(f"Hiyerarşik Kümeleme: '{feature_names[idx_cholesterol]}' vs '{feature_names[idx_oldpeak]}'")
plt.xlabel(feature_names[idx_cholesterol] + " (Ölçeklenmiş)")
plt.ylabel(feature_names[idx_oldpeak] + " (Ölçeklenmiş)")
plt.colorbar(label='Küme Etiketi', ticks=range(n_clusters))
plt.grid(True, alpha=0.6)
plt.show()