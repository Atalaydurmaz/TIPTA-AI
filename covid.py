# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 21:29:43 2025

@author: Fatma
"""

import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Yapay Tıbbi Veri Seti Oluşturma (COVID-19 Benzeri Fenotipler)
# Üç ana fenotip (Küme) oluşturuyoruz: Hafif, Orta, Şiddetli

# Küme 1: Hafif Semptomlar (Genç, Düşük CRP)
np.random.seed(42)
data_mild = {
    'Age': np.random.normal(40, 5, 100),  #ortalaması 40, standart sapması 5 olan 100 rastgele yaş değeri üretir
    'SpO2': np.random.normal(96, 1, 100),
    'CRP': np.random.normal(10, 3, 100),
}
df_mild = pd.DataFrame(data_mild)

# Küme 2: Orta Semptomlar (Orta Yaş, Orta CRP)
data_mod = {
    'Age': np.random.normal(60, 8, 100),
    'SpO2': np.random.normal(92, 2, 100),
    'CRP': np.random.normal(50, 15, 100),
}
df_mod = pd.DataFrame(data_mod)

# Küme 3: Şiddetli Semptomlar (Yaşlı, Yüksek CRP)
data_sev = {
    'Age': np.random.normal(75, 7, 100),
    'SpO2': np.random.normal(88, 3, 100),
    'CRP': np.random.normal(120, 30, 100),
}
df_sev = pd.DataFrame(data_sev)

data = pd.concat([df_mild, df_mod, df_sev], ignore_index=True)

# 2. Veri Ön İşleme (Ölçeklendirme)
# GMM ve diğer mesafe tabanlı algoritmalar için zorunludur.
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# 3. GMM Uygulaması (Yumuşak Kümeleme)
# n_components=3 (3 fenotip bekliyoruz: Hafif, Orta, Şiddetli)
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(data_scaled)

# Her nokta için hangi kümeye ait olma olasılığını hesaplama (Yumuşak Atama)
# Bu, GMM'in kalbidir.
probabilities = gmm.predict_proba(data_scaled) 
data['Prob_Cluster_1'] = probabilities[:, 0]
data['Prob_Cluster_2'] = probabilities[:, 1]
data['Prob_Cluster_3'] = probabilities[:, 2]

# En yüksek olasılığa göre 'sert' bir etiket atama (sadece görselleştirme ve özet için)
data['GMM_Cluster_Label'] = gmm.predict(data_scaled)

# 4. Analiz ve Yorumlama

print("--- Yumuşak Kümeleme (GMM) Sonuçlarından Örnekler ---")
print("Seçilen hastaların farklı kümelere ait olma olasılıkları:")
# Sınırda olan hastaları görmek için rastgele birkaç satır gösterelim
print(data[['Prob_Cluster_1', 'Prob_Cluster_2', 'Prob_Cluster_3', 'GMM_Cluster_Label']].sample(5))

print("\n--- GMM Fenotiplerinin Klinik Özellikleri ---")
# Her kümenin ortalama klinik özelliklerini hesaplayalım
cluster_summary = data.groupby('GMM_Cluster_Label')[['Age', 'SpO2', 'CRP']].mean()
print(cluster_summary)

# 5. Görselleştirme (CRP vs SpO2)
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='CRP', 
    y='SpO2', 
    hue='GMM_Cluster_Label', 
    palette='viridis', 
    s=100, 
    data=data,
    legend='full'
)
plt.title('GMM ile COVID-19 Fenotipleri (CRP vs Oksijen Saturasyonu)')
plt.xlabel('C-Reaktif Protein (CRP) Seviyesi (İltihap)')
plt.ylabel('Oksijen Saturasyonu (SpO2)')
plt.legend(title='Fenotip')
plt.grid(True, alpha=0.5)
plt.show()