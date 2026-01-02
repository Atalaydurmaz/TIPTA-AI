# -*- coding: utf-8 -*-
"""
REINFORCE (Politika Gradyanı) algoritması ile CartPole-v1 ortamını eğitir.
GÖRSELLEŞTİRME: Ortamın penceresini açarak arabanın hareketlerini izler.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import gymnasium as gym
import matplotlib.pyplot as plt

# ---------------- 1️⃣ Ortam ve Model Tanımlamaları ----------------

# Gymnasium Ortamını GÖRÜNTÜLEME MODU (render_mode="human") ile başlatma
# Bu, eğitimi izlerken simülasyon penceresinin açılmasını sağlar.
env = gym.make("CartPole-v1", render_mode="human")
state_size = env.observation_space.shape[0] # 4 boyutlu durum
action_size = env.action_space.n            # 2 eylem
LEARNING_RATE = 0.001
GAMMA = 0.99                                # Monte Carlo Getirisinde önemli

# ---------------- 2️⃣ Politika Ağı (Policy Network) ----------------
def build_policy_model():
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=(state_size,)))
    model.add(Dense(32, activation='relu'))
    # Çıktı: Her eylem için olasılık dağılımı (Softmax)
    model.add(Dense(action_size, activation='softmax'))
    return model

policy_model = build_policy_model()

# ---------------- 3️⃣ REINFORCE Kayıp Fonksiyonu ----------------

def reinforce_loss(y_true, y_pred):
    # y_true (Gerçek Değerler): [seçilen_eylem_indeksi, kümülatif_getiri (G)]
    
    reward = y_true[:, 1] #Her örneğin ödülünü alır.
    action_index = tf.cast(y_true[:, 0], tf.int32) #Her örnekte seçilen eylem indeksini alır.
    
    # Seçilen eylemin olasılığını bul
    batch_indices = tf.range(tf.shape(y_pred)[0])
    indices = tf.stack([batch_indices, action_index], axis=1)
    probabilities = tf.gather_nd(y_pred, indices)
    
    # REINFORCE Kayıp: -log(pi(a|s)) * G
    log_probabilities = K.log(probabilities + K.epsilon())
    loss = - log_probabilities * reward
    
    return K.mean(loss)

policy_model.compile(loss=reinforce_loss, optimizer=Adam(learning_rate=LEARNING_RATE))

# ---------------- 4️⃣ Kümülatif Getiri Hesaplama Fonksiyonu ----------------
# Gt = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...
def compute_returns(rewards, gamma):
    returns = []
    G = 0
    # Ödülleri sondan başa doğru toplayarak Getiriyi hesapla
    for r in rewards[::-1]:
        G = r + gamma * G
        returns.insert(0, G)
    return np.array(returns)

# ---------------- 5️⃣ Eğitim Döngüsü ----------------
N_EPISODES = 150 # Görselleştirdiğimiz için daha kısa tuttuk.
MAX_STEPS = 500  # Bir epizotun maksimum uzunluğu
score_history = []

print("\n🎯 REINFORCE (CartPole) eğitimi başlatıldı ve görselleştiriliyor...")
print("Eğitim sırasında CartPole penceresini göreceksiniz. (Kodu durdurmak için kapatabilirsiniz)")

for ep in range(1, N_EPISODES + 1):
    
    # Her epizot için verileri sıfırla
    states, actions, rewards = [], [], []
    
    # 1. Epizotu Çalıştır ve Deneyimi Topla
    state, _ = env.reset()
    done = False
    
    while not done and len(states) < MAX_STEPS:
        
        # State'i ağa uygun boyuta getir: (1, 4)
        state_input = state.reshape(1, state_size)
        
        # Politika Ağını Kullanarak Eylem Olasılıklarını Tahmin Et
        action_probs = policy_model.predict(state_input, verbose=0)[0]
        
        # Olasılıklara Göre Rastgele Eylem Seç
        action_idx = np.random.choice(action_size, p=action_probs)
        
        # Eylemi Gerçekleştir
        next_state, reward, done, truncated, _ = env.step(action_idx)
        
        # 🌟 GÖRSELLEŞTİRME KOMUTU 🌟
        # Bu satır, her adımda CartPole simülasyonunu güncelleyip ekranda gösterir.
        env.render() 
        
        # Deneyimi Kaydet
        states.append(state)
        actions.append(action_idx)
        rewards.append(reward)
        state = next_state
    
    # Toplam Skoru Kaydet
    total_reward = sum(rewards)
    score_history.append(total_reward)
    
    # 2. Kümülatif Getiriyi Hesapla
    returns = compute_returns(rewards, GAMMA)
    
    # 3. Eğitimi Hazırla ve Ağı Güncelle
    target_output = np.stack([actions, returns], axis=1)
    
    policy_model.fit(
        np.array(states),
        target_output,
        epochs=1,
        verbose=0,
        shuffle=False
    )
    
    # Konsol Çıktısı
    if ep % 50 == 0:
        print(f"Epizot: {ep}/{N_EPISODES}, Son 50 Ort. Puan: {np.mean(score_history[-50:]):.2f}")

# ---------------- 6️⃣ Ortamı Kapat ----------------
# Eğitim bittiğinde açılan pencereyi kapatır.
env.close()

# ---------------- 7️⃣ Grafik ----------------
window_size = 50
plt.figure(figsize=(12, 6))
plt.plot(np.convolve(score_history, np.ones(window_size)/window_size, mode='valid'))
plt.title("REINFORCE (CartPole-v1): 50-Epizotluk Hareketli Ortalama Puan")
plt.xlabel("Epizot")
plt.ylabel("Ortalama Puan (Direği Dik Tutma Süresi)")
plt.axhline(y=475, color='r', linestyle='--', label='Çözülmüş Eşik (475)')
plt.legend()
plt.show()