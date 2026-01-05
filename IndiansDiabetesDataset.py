# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 21:31:11 2025

@author: Fatma
"""

import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer 
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Veri Yükleme (UCI Pima Indians Diabetes Dataset)
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
           'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
data = pd.read_csv(url, names=columns)

# 2. Ön İşleme ve Eksik Veri Yönetimi
cols_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
data[cols_to_impute] = data[cols_to_impute].replace(0, np.nan)
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
data[cols_to_impute] = imputer.fit_transform(data[cols_to_impute])

X = data.drop('Outcome', axis=1)

# Veriyi Ölçeklendirme (GMM için zorunludur)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. GMM Uygulaması (Yumuşak Kümeleme)
n_clusters = 3
gmm = GaussianMixture(n_components=n_clusters, random_state=42)
gmm.fit(X_scaled)

# Olasılıkları ve sert etiketi ekleme
probabilities = gmm.predict_proba(X_scaled) 
for i in range(n_clusters):
    data[f'Prob_Cluster_{i}'] = probabilities[:, i]
data['GMM_Cluster_Label'] = gmm.predict(X_scaled).astype(str) # Görselleştirme için string yapalım

# GMM merkezlerini hesaplama (Scaled uzaydan orijinal uzaya geri dönüştürme)
# Bu merkezler, GMM'in bulduğu her Gauss dağılımının ortalamasını temsil eder.
gmm_centers_scaled = gmm.means_
gmm_centers_original = scaler.inverse_transform(gmm_centers_scaled)
centers_df = pd.DataFrame(gmm_centers_original, columns=X.columns)
centers_df['Cluster'] = centers_df.index.astype(str)


# 4. Saçılım Grafiği (Scatter Plot) Ekleme: Glikoz ve BMI Üzerinden Dağılım

plt.figure(figsize=(10, 7))

# Veri noktalarını GMM tarafından atanan sert etiketlere göre renklendirme
sns.scatterplot(
    x='Glucose', 
    y='BMI', 
    hue='GMM_Cluster_Label', 
    palette='Set1', 
    data=data,
    s=70,  # Nokta boyutu
    alpha=0.7, # Saydamlık, örtüşmeyi görmeye yardımcı olur
    legend='full'
)

# Küme Merkezlerini (Mean) İşaretleme
plt.scatter(
    centers_df['Glucose'], 
    centers_df['BMI'], 
    marker='X', # Merkezleri X işaretiyle göster
    s=300, 
    color='black', 
    linewidths=2,
    label='Küme Merkezleri (GMM Means)'
)

plt.title('GMM Yumuşak Kümeleme Sonucu: Glikoz ve BMI Dağılımı')
plt.xlabel('Glikoz Konsantrasyonu (Ortalama Risk Faktörü)')
plt.ylabel('BMI (Vücut Kitle İndeksi)')
plt.legend(title='Fenotip')
plt.grid(True, alpha=0.4)
plt.show()

# Önceki analiz çıktılarını da tekrar gösterelim:
print("--- GMM Fenotiplerinin Klinik Özellikleri (Ortalamalar) ---")
cluster_summary = data.groupby('GMM_Cluster_Label')[['Glucose', 'BMI', 'Age', 'Outcome']].mean()
print(cluster_summary.rename(columns={'Outcome': 'Diyabet Oranı (Outcome=1)'}))