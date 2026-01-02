# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 20:18:47 2025

@author: Fatma
"""

# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer

# ---------------- 1️⃣ Veri Yükleme ----------------
print("📥 UCI COVID-19 verisi yükleniyor...")
# Lütfen KENDİ DOSYA YOLUNUZU BURAYA YAZIN
file_path = r"C://Users//Fatma//Desktop//V4//covid_50.csv" 
try:
    df = pd.read_csv(file_path)
    print("✅ Veri başarıyla yüklendi!")
except FileNotFoundError:
    print("HATA: Dosya yolu bulunamadı. Lütfen dosya yolunu kontrol edin.")
    exit()

# ---------------- 2️⃣ Ön işleme ----------------
target_col = 'COVID-19'

# Kategorikleri sayıya çevir
for c in df.columns:
    if df[c].dtype == object:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))#METİN tabanlı tüm 
        #sütunları 0,1,2 gibi tamsayı değerlerine çevirir.

X = df.drop(columns=[target_col])
y = df[target_col]

# Eksik veri doldur
X = X.fillna(0)#Eksik değerler 0 ile doldurulur
y = y.fillna(0)

# ---------------- 3️⃣ Diskritizasyon ve State Temsili ----------------
N_BINS = 2 #Her sayısal özelliği iki kateoriye ayırmayı seçer
kbd = KBinsDiscretizer(n_bins=N_BINS, encode='ordinal', strategy='uniform')#Veriyi
#öğrenilen NBINS sayısına göre bölüyor
X_disc_all = kbd.fit_transform(X)#tüm sayısal özellikleri 0 veya 1 gibi ayrık değerlere dönüştürür

# Diskritize edilmiş veriyi kullanacağız
X_train_disc, X_test_disc, y_train, y_test = train_test_split(
    X_disc_all, y, test_size=0.2, random_state=42
)

# Durumları temsil eden özellik sayısı
n_features = X_train_disc.shape[1]
actions = [0, 1, 2]  # 0: no action, 1: test, 2: treat
n_actions = len(actions)

print(f"\nModel Girdi Boyutu (Özellik Sayısı): {n_features}")
print(f"Eylem sayısı: {n_actions}")

# ---------------- 4️⃣ Ödül fonksiyonu ----------------
def reward_fn(action, true_label):
    if action == 0:  # hiçbir şey yapma
        # true=0 ise ödül 0, true=1 ise ceza -1
        return 0 if true_label == 0 else -1 
    if action == 1:  # test et
        # true=1 ise ödül 1, true=0 ise test maliyeti -0.2
        return 1 if true_label == 1 else -0.2
    if action == 2:  # tedavi et
        # true=1 ise ödül 1, true=0 ise yüksek ceza -1 (yanlış tedavi)
        return 1 if true_label == 1 else -1
    return 0

# ---------------- 5️⃣ Q-TABLOSU (DQN Yerine) ----------------
# Durumların HASH'lenmesi: Q-tablosu için her ayrık durumun benzersiz bir indeksini oluştururuz.
# Bu, (0, 1, 0, 1, ...) gibi ayrık özellikleri tek bir tam sayıya çevirir.
def state_to_index(s_disc):
    # s_disc: (n_features,) boyutunda numpy dizisi (örneğin: [0. 1. 0. 1. ...])
    s_tuple = tuple(s_disc.astype(int))
    return hash(s_tuple)
#İşleyiş Adımları
#Giriş: Fonksiyon, s_disc adı verilen, ayrıklaştırılmış (discrete) durumu temsil eden bir NumPy dizisini alır.
# (Örnek: [0. 1. 0. 1. ...]) Tamsayıya Dönüştürme:s_disc.astype(int): Dizideki tüm kayan nokta (float) değerleri, tamsayılara dönüştürülür 
#Bu, tutarlı bir hash değeri üretmek için önemlidir.Tuple'a Dönüştürme:tuple(...): Tamsayılardan oluşan NumPy dizisi, tuple (demet) veri yapısına çevrilir.
#Neden Tuple? Python'da yalnızca değişmez (immutable) nesneler hash'lenebilir ve sözlük anahtarı olarak kullanılabilir. 
#NumPy dizileri ve listeler değişken (mutable) olduğu için doğrudan anahtar olamazlar; ancak tuple'lar değişmezdir.
#Hashleme:hash(s_tuple): Elde edilen tuple, Python'ın yerleşik hash() fonksiyonu kullanılarak tek, benzersiz bir tamsayıya (integer) dönüştürülür.
# Pekiştirmeli Öğrenme'deki RolüBu hash değeri, daha sonra Q-Tablosu'nda o duruma karşılık gelen Q-değerlerini depolamak için anahtar (key) olarak kullanılır. 
#Bu sayede, ajan o durumu tekrar ziyaret ettiğinde, hızlıca ilgili Q-değerlerine erişebilir.

# Tüm veri setindeki benzersiz durumları bulalım
unique_states = {state_to_index(s) for s in X_train_disc}
#Bu satır, eğitim veri setinde (training set) bulunan tüm benzersiz durumları (states) toplar
# ve her bir durumu benzersiz bir anahtara (hash) dönüştürür.

Q_table = {state_hash: np.zeros(n_actions) for state_hash in unique_states}
#Bu satır, unique_states kümesindeki her bir benzersiz durum anahtarı için Q-Tablosu'nu oluşturur 
#ve başlangıç Q-değerlerini sıfır olarak ayarlar.

print(f"Q-Tablosu Boyutu (Benzersiz Durum Sayısı): {len(Q_table)}")

# ---------------- 6️⃣ Q-Learning Parametreleri ----------------
ALPHA = 0.5  # Öğrenme Oranı (ALPHA) #Amaç: Ajanın yeni bilgiyi ne kadar ciddiye alacağını belirler.
GAMMA = 0.9  # İndirgeme Faktörü (Gelecekteki ödüllerin ne kadar önemli olduğunu belirler.)
EPS_START = 1.0 #Öğrenmenin başlangıcında keşif yapma olasılığı. Genellikle %100 (1.0) başlar, yani ajan en başta rastgele davranır.
EPS_END = 0.01 #Öğrenmenin sonunda keşif yapma olasılığının düşeceği minimum değer.
EPS_DECAY = 0.99977 #eps değerinin her adımda ne kadar azaltılacağını (çarpılacağını) belirler.
N_EPISODES = 20000  #Ajanın toplamda kaç farklı "oyun" veya "deneme" döngüsü (epizot) yapacağını belirler.

eps = EPS_START
rewards_history = []
print("\n🎯 Q-Learning başlatıldı...")

# ---------------- 7️⃣ Eğitim Döngüsü (Q-Learning Mantığı) ----------------
for ep in range(1, N_EPISODES + 1):#20.000 defa eğitim epizotu başlatılır.
    # Rastgele bir eğitim örneği seç (durum/satır)  1. Rastgele eğitim örneği seçmek
    idx = np.random.randint(len(X_train_disc))#Eğitim veri setinden rastgele bir hasta (durum) seçilir.
    s_disc = X_train_disc[idx]#seçilen durum alınır
    true_label = y_train.iloc[idx]#COVID-19'un gerçekte olup olmadığı) alınır.
    
    # Durumu hash'e çevir
    s_hash = state_to_index(s_disc)#Seçilen durum, $Q$-Tablosu'nda anahtar olarak kullanılacak benzersiz bir hash değerine dönüştürülür.
    
    # 7.1. Epsilon-greedy (Keşif)  2. Epsilon-greedy ile rastgele eylem seçmek (keşif)
    if random.random() < eps:#eğer rastgele üretilen sayı mevcut eps değerinden küçükse
        a_idx = random.randrange(n_actions) #ajan rastgele bir eylem seçer
    else:
        #  Aksi takdirde, ajan Q-Tablosu'na bakar ve mevcut durumda en yüksek beklenen ödüle sahip olan eylemi seçerek öğrendiği bilgiyi sömürür.
        a_idx = np.argmax(Q_table[s_hash]) 

    a = actions[a_idx] #daha önce belirlenen eylem indeksini (a_idx), ajan tarafından gerçekleştirilecek gerçek eyleme (a) dönüştürür.
    r = reward_fn(a, true_label) #ajanın amacını belirleyen kritik fonksiyondur.
    #Girdi 1 (a): Ajanın tahmini sonucu veya en uygun gördüğü eylemi yansıtır.
    #Örneğin, ajan hastanın özelliklerine bakarak COVID-19 olma ihtimalinin yüksek olduğunu tahmin eder ve buna dayanarak Tedavi Et (a=2) eylemini seçer.
    #Girdi 2 (true_label): Seçilen hastanın gerçek COVID-19 durumu (Örn: 1 - COVID var veya 0 - COVID yok).
    #Çıktı (r): Ajanın bu eylem-sonuç çifti için aldığı sayısal değer (ödül >0 veya ceza <0).
    
    # 7.2. Q-Tablosu Güncellemesi (Bellman Denklemi)
  #Bu kod bloğu, Q-Learning Algoritmasının Kalbini, yani Q-Tablosu'nu güncelleme işlemini gerçekleştirir. 
  #Bu güncelleme, ajanın deneyimlerinden öğrenmesini sağlayan Bellman Denklemi'nin uygulanmasıdır.Q-Tablosu Güncelleme Adımları
  #Bu kısım, ajanın bir eylem yaptıktan sonra kazandığı ödülü kullanarak, mevcut durum-eylem çifti için tahminini nasıl düzelttiğini gösterir.  
    # Sonraki durumu simüle et (yine rastgele bir sonraki durum çekiyoruz)
    idx_next = np.random.randint(len(X_train_disc))#Eğitim veri setinden rastgele bir hasta (durum) seçilir.
    s_next_disc = X_train_disc[idx_next]#seçilen durum alınır
    s_next_hash = state_to_index(s_next_disc)#Seçilen durum, Q-Tablosu'nda anahtar olarak kullanılacak benzersiz bir hash değerine dönüştürülür.

    # Sonraki durumun Q-Tablosundaki en yüksek değerini bul
    Q_next_max = np.max(Q_table[s_next_hash])#Ajan, mevcut eyleminin faydasını değerlendirirken,
    #bu eylem sonucu geçeceği yeni durumdan (s') gelecekte alabileceği en iyi potansiyel ödülü hesaba katmak zorundadır.

    # Güncelleme için Bellman Denklemi
    # Q(s, a) = Q(s, a) + ALPHA * [r + GAMMA * max(Q(s', a')) - Q(s, a)]
    old_q = Q_table[s_hash][a_idx]#Mevcut durum ($s$) ve eylem ($a$) için ajanın eski tahminidir.
    new_q_value = old_q + ALPHA * (r + GAMMA * Q_next_max - old_q)#Ajan için yeni bilgi (hedef) veya düzeltilmiş ödüldür.
    
    # Q-Tablosunu güncelle
    Q_table[s_hash][a_idx] = new_q_value
    
    rewards_history.append(r)#Ajanın o anki eyleminden dolayı elde ettiği anlık ödülü (r),
    #eğitim sürecindeki ödüllerin tüm geçmişini tutan rewards_history listesine ekler.
    eps = max(EPS_END, eps * EPS_DECAY)

    if ep % 2000 == 0:
        print(f"Epizot: {ep}/{N_EPISODES}, Epsilon: {eps:.4f}")

# ---------------- 8️⃣ Test aşaması ----------------
correct, total, cum_reward = 0, 0, 0
#correct: Ajanın doğru kabul edilen kararlarının sayısını tutar.
#İşlenen toplam test örneği sayısını (hasta sayısını) tutar.
#Test aşaması boyunca ajanın topladığı kümülatif (birikmiş) ödülü tutar.
print("\n🔬 Test aşaması başlatıldı...")
##Döngü: Kod, test durumları (X_test_disc) ve bu durumların gerçek etiketleri (y_test) üzerinde eş zamanlı olarak döner.
for s_disc, true in zip(X_test_disc, y_test):#
    s_hash = state_to_index(s_disc)#Her bir test durumu, eğitimde kullanılan state_to_index fonksiyonu ile hash değerine dönüştürülür.
#not:Q-Tablosu yalnızca eğitim aşamasında ajanın karşılaştığı benzersiz durumları içerir.    
    # Q-Tablosunda durum varsa en iyi eylemi seç, yoksa 0 (no action) seç
    if s_hash in Q_table:
        a_idx = np.argmax(Q_table[s_hash])
        a = actions[a_idx]
    else:
        # Bu durum eğitimde görülmemiştir, varsayılan olarak "Hiçbir şey yapma"
        a = 0 
#Eğer durumun hash'i Q-Tablosu'nda varsa, ajan o durum için öğrendiği Q-değerlerine bakar 
#ve np.argmax ile en yüksek değere sahip olan eylemi (a) seçer.
#Eğer durum eğitimde hiç görülmemişse (yani hash Q-Tablosu'nda yoksa), ajan risk almamak için 
#varsayılan olarak "Hiçbir şey yapma" (a = 0) eylemini seçer.
    r = reward_fn(a, true)#Seçilen eylem (a) ve hastanın gerçek durumu (true) kullanılarak ÖDÜL HESAPLANIR
    
    cum_reward += r#Hesaplanan ödül cum_reward'a eklenir. Bu, ajanın genel ekonomik başarısını ölçer.
    total += 1#Bu, test veri setindeki işlenen toplam hasta (durum) sayısını bir artırır. 
    #Bu değişken, sonunda doğruluk oranını (correct/total) hesaplamak için kullanılır.
    
    # Metrik: Eğer COVID-19 ise test/tedavi (1, 2) yapıldıysa VEYA COVID-19 değilse hiçbir şey yapma (0) yapıldıysa doğru kabul et.
    if (true == 1 and a in [1, 2]) or (true == 0 and a == 0):
        correct += 1
#true == 1 and a in [1, 2]
#Anlamı: Hastada gerçekte COVID-19 Varsa (true == 1), ajanın eylemi Test Et (1) veya Tedavi Et (2) olmalıdır.
#true == 0 and a == 0
#Hastada gerçekte COVID-19 Yoksa (true == 0), ajanın eylemi Hiçbir şey yapma (0) olmalıdır.


print(f"\n✅ Test doğruluk (oyuncak metrik): {correct/total:.3f}")
print(f"🎁 Toplam ödül: {cum_reward:.2f}")

# ---------------- 9️⃣ Grafik ----------------
window_size = 500 # Daha uzun bir pencere ile hareketli ortalama
plt.figure(figsize=(12, 6))
plt.plot(np.convolve(rewards_history, np.ones(window_size)/window_size, mode='valid'))
plt.title("Q-Learning COVID: 500-adımlık Hareketli Ortalama Ödül")
plt.xlabel("Eğitim Adımı")
plt.ylabel("Ortalama Ödül")
plt.show()