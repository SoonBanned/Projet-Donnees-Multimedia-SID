import os
import subprocess

# Dossier contenant les vidéos
input_folder = "./videos"
# Dossier où tu veux sauvegarder les fichiers audio
output_folder = "./audios"

# Créer le dossier de sortie s’il n’existe pas
os.makedirs(output_folder, exist_ok=True)

# Formats vidéo pris en charge
video_extensions = (".mp4", ".mov", ".avi", ".mkv", ".flv")

# Parcourir toutes les vidéos du dossier
for filename in os.listdir(input_folder):
    if filename.lower().endswith(video_extensions):
        video_path = os.path.join(input_folder, filename)
        output_path = os.path.join(
            output_folder, os.path.splitext(filename)[0] + ".wav"
        )

        print(f"🎬 Extraction de l’audio depuis : {filename}")

        # Commande FFmpeg : extraire l’audio en WAV (PCM non compressé)
        command = [
            "ffmpeg",
            "-i",
            video_path,  # fichier vidéo en entrée
            "-vn",  # pas de vidéo
            "-acodec",
            "pcm_s16le",  # codec WAV non compressé
            "-ar",
            "44100",  # fréquence d’échantillonnage (44.1 kHz)
            "-ac",
            "2",  # stéréo
            output_path,
            "-y",  # écrase si déjà existant
        ]

        # Exécuter la commande sans afficher la sortie FFmpeg
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

print("✅ Extraction terminée ! Tous les fichiers WAV sont dans :", output_folder)
