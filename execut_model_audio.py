import os
import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import soundfile as sf
import scipy.signal
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# === 1. Charger le modèle personnalisé ===
custom_model = tf.keras.models.load_model("yamnet_custom_classifier.h5")

# === 2. Charger YAMNet ===
yamnet_model = hub.load('https://www.kaggle.com/models/google/yamnet/TensorFlow2/yamnet/1')
# === 3. Fonctions utilitaires ===
def ensure_sample_rate(original_sr, waveform, desired_sr=16000):
    """Rééchantillonne le signal si besoin."""
    if original_sr != desired_sr:
        desired_len = int(round(len(waveform) * desired_sr / original_sr))
        waveform = scipy.signal.resample(waveform, desired_len)
    return desired_sr, waveform

def extract_embedding(wav_path):
    """Extrait l’embedding YAMNet moyen pour un fichier audio donné."""
    wav_data, sr = sf.read(wav_path)
    if wav_data.ndim > 1:
        wav_data = np.mean(wav_data, axis=1)
    sr, wav_data = ensure_sample_rate(sr, wav_data)
    _, embeddings, _ = yamnet_model(wav_data)
    return np.mean(embeddings.numpy(), axis=0)

# === 4. Charger les catégories ===
category_map = {}
with open("train_val_annotation/category.txt") as f:
    for line in f:
        name, idx = line.strip().split("\t")
        category_map[int(idx)] = name
inv_category_map = {v: k for k, v in category_map.items()}

# === 5. Charger le JSON de test ===
with open("test_videodatainfo.json/test_videodatainfo.json") as f:
    test_data = json.load(f)

# === 6. Extraire les embeddings et labels ===
X_test, y_test, missing_files = [], [], []

for vid in test_data["videos"]:
    # 🧠 Vérifie le bon champ dans ton JSON :
    # Parfois c’est "video_id", parfois juste "id"
    video_id = vid.get("video_id", vid.get("id"))

    # Chemin du fichier audio
    wav_path = os.path.join("test_val_audios", f"{video_id}.wav")

    if os.path.exists(wav_path):
        emb = extract_embedding(wav_path)
        X_test.append(emb)
        y_test.append(vid["category"])
    else:
        missing_files.append(wav_path)

# === 7. Vérification avant la prédiction ===
print(f"\n✅ Fichiers audio trouvés : {len(X_test)}")
print(f"❌ Fichiers manquants : {len(missing_files)}")

if missing_files:
    print("\nExemples de fichiers manquants :")
    print("\n".join(missing_files[:5]))

if len(X_test) == 0:
    raise ValueError("Aucun fichier audio trouvé — vérifie les chemins et noms de fichiers .wav")

# === 8. Prédiction ===
X_test = np.array(X_test)
y_test = np.array(y_test)

y_pred = custom_model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)

# === 9. Rapport de performance ===
print("\n=== Rapport de classification ===")
print(classification_report(y_test, y_pred_labels,
      target_names=[category_map[i] for i in sorted(category_map.keys())]))

# === 10. Matrice de confusion ===
cm = confusion_matrix(y_test, y_pred_labels)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[category_map[i] for i in sorted(category_map.keys())],
            yticklabels=[category_map[i] for i in sorted(category_map.keys())])
plt.xlabel("Catégorie prédite")
plt.ylabel("Catégorie réelle")
plt.title("Matrice de confusion — Modèle YAMNet personnalisé")
plt.show()
