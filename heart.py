import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# (1) Sentetik Veri Oluşturma ve Hazırlama
# ==========================================================
np.random.seed(42)
data_size = 303 
X = np.zeros((data_size, 5))

# Kalp Hastalığı Veri Setine Benzer Özellikler
X[:, 0] = np.random.normal(54, 8, data_size)   # Yaş (age)
X[:, 1] = np.random.randint(1, 5, data_size)   # Göğüs Ağrısı Tipi (chest_pain_type)
X[:, 2] = np.random.normal(245, 50, data_size) # Kolesterol (cholesterol)
X[:, 3] = np.random.normal(150, 20, data_size) # Maks. Kalp Atış Hızı (max_heart_rate)
X[:, 4] = np.random.uniform(0, 4, data_size)   # oldpeak (Egzersize bağlı ST depresyonu)

feature_names = ['age', 'chest_pain_type', 'cholesterol', 'max_heart_rate', 'oldpeak']

# Veri Standardizasyonu (Ölçekleme)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# (2) Hiyerarşik Kümeleme Modelini Oluşturma ve Eğitme
# ==========================================================
n_clusters = 4 

model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')

# Kümeleme işlemini gerçekleştir
model.fit(X_scaled)
labels = model.labels_

# (3) Dendrogram Çizimi ve Eşik Ekleme
# ==========================================================
# 'ward' linkage metodu ile birleşme matrisini (Z) oluştur
Z = linkage(X_scaled, method='ward')

# 4 küme elde etmek için görseldeki en uygun eşik mesafesini belirliyoruz (~15.0)
KESME_MESAFESI = 14

plt.figure(figsize=(15, 7))
dendrogram(
    Z,
    truncate_mode='lastp',
    p=30, # Son 30 birleşmeyi göster (daha okunur olması için)
    leaf_rotation=90.,
    leaf_font_size=8.,
    show_contracted=True,
    # Bu eşiğin altındaki dallar farklı renklerde gösterilecektir:
    color_threshold=KESME_MESAFESI
)

# Eşik çizgisini ekleyerek 4 kümenin ayrıldığı seviyeyi görselleştirme
plt.axhline(y=KESME_MESAFESI, c='black', linestyle='--', linewidth=2, label=f'4 Küme Eşiği ({KESME_MESAFESI} Mesafesi)')
plt.legend()

plt.title('Kalp Hastalığı Veri Seti Hiyerarşik Kümeleme Dendrogramı (Eşik Çizgisiyle)')
plt.xlabel(f'Hasta İndeksi veya Birleşen Küme Sayısı (Toplam {len(X)})')
plt.ylabel('Mesafe')
plt.show()

# (4) Sonuçları Görselleştirme (2 Boyutta)
# ==========================================================
# İki önemli özelliği seçelim: 'age' ve 'max_heart_rate'
idx_age = feature_names.index('age')
idx_thalach = feature_names.index('max_heart_rate')

plt.figure(figsize=(10, 6))
plt.scatter(X_scaled[:, idx_age], X_scaled[:, idx_thalach], c=labels, cmap='Spectral', 
            s=50, edgecolors='k')


#X ve Y Eksenleri: Grafiğin yerleşimi (noktaların konumu) sadece 2 özelliğe (age ve max_heart_rate) göre belirlenir.

#Renkler (c=labels): Noktaların rengi ise, modelin 5 özellikten elde ettiği 5 farklı küme etiketine göre atanır.


plt.title(f"Hiyerarşik Kümeleme: '{feature_names[idx_age]}' vs '{feature_names[idx_thalach]}'")
plt.xlabel(feature_names[idx_age] + " (Ölçeklenmiş)")
plt.ylabel(feature_names[idx_thalach] + " (Ölçeklenmiş)")
plt.colorbar(label='Küme Etiketi', ticks=range(n_clusters))
plt.grid(True, alpha=0.6)
plt.show()