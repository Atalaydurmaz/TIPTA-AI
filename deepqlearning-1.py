# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# ESKİ SATIR (Hata Veren): from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.optimizers import Adam # ✅ Yeni ve Keras 3 ile uyumlu Adam importu
from sklearn.preprocessing import KBinsDiscretizer

# ---------------- 1️⃣ Veri setini internetten çek ----------------
# Veri yolu güncellendi
print("📥 UCI COVID-19 verisi indiriliyor...")
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
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))

X = df.drop(columns=[target_col])
y = df[target_col]

# Eksik veri doldur
X = X.fillna(0)
y = y.fillna(0)

# ---------------- 3️⃣ Diskritizasyon ve State Temsili ----------------
# Q-Learning'deki aşırı büyük durum sayısını çözmek için diskitizasyon seviyesi korunmuştur.
# Ancak DQN'de bu durumlar doğrudan Sinir Ağına GİRDİ olarak verilecektir, HASH KULLANILMAYACAKTIR.
N_BINS = 2 #☻o anda verideki sayısal çeşitliliği 2 ye ayırırsınız
kbd = KBinsDiscretizer(n_bins=N_BINS, encode='ordinal', strategy='uniform')
#KBinsDiscretizer, popüler makine öğrenimi kütüphanesi scikit-learn'den gelen bir araçtır. Amacı, veri kümesindeki her bir sürekli sayısal özelliği alıp, o değerleri belirli sayıda eşit parçaya (kutucuğa veya kategoriye) ayırmaktır.
X_disc_all = kbd.fit_transform(X) # Basitçe: "Böleceğim sınırlamaları öğren ve hemen tüm veriyi 0 veya 1 olarak kategorilendir."

# Diskritize edilmiş (ayrıklaştırılmış) veriyi kullanacağız, bu da durum uzayının boyutunu azaltır.
X_train_disc, X_test_disc, y_train, y_test = train_test_split(
    X_disc_all, y, test_size=0.2, random_state=42
)

n_features = X_train_disc.shape[1]# Bu, modelin bir hastanın durumunu tanımlayan özellik sayısıdır (örneğin, $21$ semptom ve risk faktörü). Sinir Ağınız bu $21$ özelliği girdi olarak alacaktır.
actions = [0, 1, 2]  # 0: no action, 1: test, 2: treat
#Anlamı: Ajanın o anda alabileceği tüm muhtemel eylemleri (kararları) tanımlar.Açıklama: 
#Bu, bir hastanın durumuna yanıt olarak alınabilecek üç farklı karardır:0: Hiçbir şey yapma1: COVID-19 testi yap2: Tedavi etRolü: 
#Bu liste, RL ajanınızın eylem uzayını oluşturur. Ajanın amacı, her durumda bu eylemlerden en yüksek $Q$ değerine sahip olanı seçmektir.
n_actions = len(actions)

print(f"\nModel Girdi Boyutu (Özellik Sayısı): {n_features}")
print(f"Eylem sayısı: {n_actions}")

# ---------------- 4️⃣ Ödül fonksiyonu ----------------
def reward_fn(action, true_label):
    if action == 0:  # hiçbir şey yapma
        return 0 if true_label == 0 else -1
    if action == 1:  # test et
        return 1 if true_label == 1 else -0.2
    if action == 2:  # tedavi et
        return 1 if true_label == 1 else -1
    return 0

# ---------------- 5️⃣ DQN Modeli (Q-Tablosu Yerine) ----------------
def build_dqn_model(input_shape, output_shape):
    model = Sequential()
    # Girdi katmanı, özellik sayısı kadar nöron alır
    model.add(Dense(32, activation='relu', input_shape=(input_shape,)))
    model.add(Dense(32, activation='relu'))
    # Çıktı katmanı, her eylem için bir Q değeri verir
    model.add(Dense(output_shape, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

# Q-Değerlerini tahmin edecek olan modelimiz
#Bu ağ, bir durum (s) verildiğinde, ajanın hangi eylemi seçeceğine karar vermek için o eylemlerin güncel Q değerlerini tahmin eder.
#öğrenme sürecini gerçekleştiren asıl sinir ağıdır.
model = build_dqn_model(n_features, n_actions)
# Hedef modeli, stabilizasyon için kullanılır (DQN'in kilit noktası)
#ana model ile aynı mimariye sahip olan, ancak farklı bir amaca hizmet edeR
#Bu ağ, Bellman Denklemi (Q-öğrenme kuralı) kullanılırken hesaplanan hedef Q değerini tahmin eder
target_model = build_dqn_model(n_features, n_actions)
target_model.set_weights(model.get_weights())

# ---------------- 6️⃣ DQN Parametreleri ----------------
ALPHA = 0.5  # RL Öğrenme Oranı (Şimdi Keras öğrenme oranını kullanacağız)
GAMMA = 0.9  # İndirgeme Faktörü (Q-Learning'de 0 idi, DQN'de 0.9 yaptık)
EPS_START = 1.0
EPS_END = 0.01 # Daha uzun keşif için biraz düşürüldü
EPS_DECAY = 0.99999 # Yavaş düşüş
N_EPISODES = 20000 # Tablo yerine ağ eğittiğimiz için daha fazla adım gerekir
BATCH_SIZE = 64
TARGET_UPDATE_FREQ = 100 # Hedef ağı kaç adımda bir güncelleyeceğimiz

eps = EPS_START
rewards_history = []
print("\n🎯 DQN başlatıldı...")

# ---------------- 7️⃣ Eğitim Döngüsü (DQN Mantığı) ----------------
# DQN, rastgele bir durumdan başlar ve her adımda ağı eğitir
for ep in range(1, N_EPISODES + 1):
    # Rastgele bir eğitim örneği seç (durum/satır)
    idx = np.random.randint(len(X_train_disc))
    s = X_train_disc[idx]
    true_label = y_train.iloc[idx]
    
    # Durumu ağa besle (bir sonraki Q değerlerini tahmin et)
    # Keras'a uygun olması için boyutu ayarla: (1, n_features)
    s_input = s.reshape(1, n_features) 
    
    # 7.1. Epsilon-greedy (Keşif)
    if random.random() < eps:
        a_idx = random.randrange(n_actions)
    else:
        # Mevcut ağdan Q değerlerini tahmin et ve en iyisini seç
        q_values = model.predict(s_input, verbose=0)[0]
        a_idx = np.argmax(q_values)

    a = actions[a_idx]
    r = reward_fn(a, true_label)
    
    # 7.2. Q-Değeri Güncellemesi (Hedef Hesaplama)
    
    # Mevcut Q değerlerini al (bu, ağın çıktısıdır)
    current_q = model.predict(s_input, verbose=0)[0]
    
    # Sonraki durum (Bu problemde, her adım bağımsız bir durumdur. Sonraki durumu simüle etmek zor.)
    # Basitleştirilmiş yaklaşımla: Sonraki durumun (s') değeri (Q_next) bu problemde sıfır alınabilir (GAMMA=0'daki gibi).
    # Ancak DQN için GAMMA'yı 0.9 aldık. Basitçe sonraki durumu aynı veri setinden rastgele çekelim.
    idx_next = np.random.randint(len(X_train_disc))
    s_next = X_train_disc[idx_next]
    s_next_input = s_next.reshape(1, n_features)

    # Hedef ağdan (target_model) sonraki Q değerini tahmin et (stabilizasyon için)
    Q_next_all = target_model.predict(s_next_input, verbose=0)[0]
    Q_next_max = np.max(Q_next_all)

    # Yeni Q hedef değeri (Bellman Denklemi)
    new_q_target = r + GAMMA * Q_next_max
    
    # YENİ HEDEF vektörünü oluştur (current_q'nun bir kopyası)
    target_f = current_q.copy()
    # Sadece seçilen eylemin Q değerini yeni hedef ile değiştir
    target_f[a_idx] = new_q_target
    
    # 7.3. Ağı Eğitme
    # Ağ, mevcut durumu girdi olarak alıp, güncellenmiş Q değerlerini (target_f) tahmin etmeye çalışır.
    model.fit(s_input, target_f.reshape(1, n_actions), epochs=1, verbose=0)
    
    rewards_history.append(r)
    eps = max(EPS_END, eps * EPS_DECAY)

    # 7.4. Hedef Ağı Güncelle
    if ep % TARGET_UPDATE_FREQ == 0:
        target_model.set_weights(model.get_weights())
        
    if ep % 1000 == 0:
         print(f"Epizot: {ep}/{N_EPISODES}, Epsilon: {eps:.4f}")


# ---------------- 8️⃣ Test aşaması ----------------
correct, total, cum_reward = 0, 0, 0

print("\n🔬 Test aşaması başlatıldı...")

for s_disc, true in zip(X_test_disc, y_test):
    s_input = s_disc.reshape(1, n_features)
    
    # Politikamız artık Q-tablosu değil, Sinir Ağıdır
    q_values = model.predict(s_input, verbose=0)[0]
    
    a_idx = np.argmax(q_values)
    a = actions[a_idx]
    r = reward_fn(a, true)
    
    cum_reward += r
    total += 1
    
    # Metrik: Eğer COVID-19 ise test/tedavi (1, 2) yapıldıysa VEYA COVID-19 değilse hiçbir şey yapma (0) yapıldıysa doğru kabul et.
    if (true == 1 and a in [1, 2]) or (true == 0 and a == 0):
        correct += 1

print(f"\n✅ Test doğruluk (oyuncak metrik): {correct/total:.3f}")
print(f"🎁 Toplam ödül: {cum_reward:.2f}")

# ---------------- 9️⃣ Grafik ----------------
window_size = 500 # Daha uzun bir pencere ile hareketli ortalama
plt.figure(figsize=(12, 6))
plt.plot(np.convolve(rewards_history, np.ones(window_size)/window_size, mode='valid'))
plt.title("DQN COVID: 500-adımlık Hareketli Ortalama Ödül")
plt.xlabel("Eğitim Adımı")
plt.ylabel("Ortalama Ödül")
plt.show()