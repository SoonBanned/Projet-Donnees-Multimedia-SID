# Projet Master 2 SID

## *Classification des données de la collection MSR-VTT*

## Introduction

Ce projet s’inscrit dans le cadre de notre projet de Master 2 qui porte sur la classification de vidéos à partir de la collection **MSR-VTT**, un jeu de données de référence en traitement de données multimédia.

### Le Jeu de données

Nous avons eu, pour projet, à notre disposition :

-   de videos divisées en ensemble de Train et Test.
-   deux fichiers *JSON* contenant chacun des informations sur chaque vidéo (1 fichier pour les Vidéos du Train et 1 fichier pour celles du Test), dont des descriptions de chaque vidéo annotées par des humains.

Les vidéos sont réparties en **20 catégories thématiques** couvrant un large éventail de contenus provenant de YouTube.

La collection est visualisable sur [Hugging face](https://huggingface.co/datasets/friedrichor/MSR-VTT). Elle est également très bien documentée dans l'article [MRS-VTT](https://openaccess.thecvf.com/content_cvpr_2016/papers/Xu_MSR-VTT_A_Large_CVPR_2016_paper.pdf).

L’objectif final est d'**implémenter et évaluer des modèles qui seront exploités conjointement pour faire de la classification directement à partir des fichiers videos** .

## Etapes du projet

Le projet s’articule autour de plusieurs étapes clés :

1.  **Analyse exploratoire des données**
    -   Statistiques descriptives : nombre de vidéos, durée moyenne, longueur moyenne des transcriptions, répartition par catégorie, etc.
2.  **Classification selon chaque modalité**
    -   *Video* : En exploitant directement les images contenues dans chaque vidéo
    -   *Audio* : En exploitant le contenu sonore extrait de chaque vidéo\
    -   *Texte* : En exploitant les descriptions contenu dans les fichiers JSON
3.  **Classification multimodale**
    -   Implémentation d’un modèle de base pour la seconde tâche,\
        en s’appuyant éventuellement sur les descripteurs extraits dans la tâche principale.

------------------------------------------------------------------------

## Méthodologie et Résultats obtenus

### Classification selon chaque modalité

Pour réaliser les différentes tâches, nous avons adopter des méthodologies différentes selon la modalité concernée.

#### 1. Vidéo

La modalité vidéo constitue le cœur du projet, permettant d’exploiter la composante visuelle et temporelle des vidéos du corpus **MSR-VTT**.

Nous avons conçu un pipeline complet d’extraction et de classification vidéo, articulé en deux étapes principales :

##### a. Extraction des caractéristiques visuelles et temporelles

L’extraction repose sur un système hybride combinant plusieurs réseaux pré-entraînés :
- **MViT (Multiscale Vision Transformer)** : capture les dépendances spatio-temporelles sur des séquences de 16 à 64 frames.
- **ResNet (frame centrale)** : extrait un embedding global à partir de la frame médiane de la vidéo.
- **Flux optique (Optical Flow – TV-L1)** : encode le mouvement entre frames via un CNN basé sur ResNet18.

Les frames sont échantillonnées de manière *uniforme* ou *aléatoire* selon la configuration, et les embeddings issus de chaque composant sont concaténés pour former un vecteur global de caractéristiques.  
Les descripteurs résultants sont sauvegardés dans le répertoire `features_finetuned/` et servent d’entrée au classifieur.

##### b. Classification vidéo

Le modèle de classification repose sur une architecture **LSTM bidirectionnelle avec couche de self-attention (SA-LSTM)**.  
L’entrée du modèle est le vecteur de caractéristiques fusionné de dimension `5632`, correspondant à la concaténation des embeddings extraits par les modèles **MViT**, **ResNet** et **Flow**.

La tête de classification comprend :
- un **LSTM à 2 couches** (dimension cachée = 384),
- une **couche d’attention** (dimension = 256),
- une régularisation par **Dropout (0.5)**, **Label smoothing (0.05)** et **MixUp (0.1)**.

L’entraînement a été effectué sur GPU avec :
- une stratégie d’**early stopping** (patience = 15),
- un **scheduler adaptatif** sur le taux d’apprentissage (facteur = 0.5, patience = 5),
- et une **pondération de classes** via `WeightedRandomSampler` pour équilibrer les données.

Les poids des modèles fine-tunés sont enregistrés sous les fichiers :  
`mvit_tuned.pt`, `resnet_tuned.pt`, `flow_tuned.pt`, `head_tuned.pt`, et le meilleur modèle SA-LSTM sous `best_sa_lstm_53,1.pt`.

Cette approche permet une **modélisation fine des dynamiques visuelles** tout en limitant le surapprentissage grâce à une régularisation contrôlée et une fusion multimodale optimisée.



#### 2. Audio

La modalité **audio** vise à exploiter la bande sonore des vidéos afin d’enrichir la classification multimodale.  
Cette étape s’articule en deux volets principaux : **l’extraction des pistes audio** et **l’apprentissage du modèle de classification sonore**.

##### a. Extraction des pistes audio

Pour chaque vidéo du corpus MSR-VTT, la piste audio est extraite à l’aide du script `extract audios.py`.  
Ce script utilise **FFmpeg** pour convertir chaque fichier vidéo (`.mp4`, `.mov`, `.avi`, `.mkv`, etc.) en un fichier **WAV** non compressé.

Les fichiers générés sont enregistrés dans le dossier `./audios/`, avec les paramètres suivants :
- fréquence d’échantillonnage : **44,1 kHz**
- nombre de canaux : **2 (stéréo)**
- format : **PCM 16 bits (pcm_s16le)**

Cette conversion garantit une qualité optimale pour l’extraction d’embeddings audio et une compatibilité totale avec les modèles basés sur TensorFlow et YAMNet.

##### b. Extraction des descripteurs et classification

L’étape suivante consiste à exploiter le contenu audio pour la classification.  
Le script `execut_model_audio.py` met en œuvre un pipeline basé sur **YAMNet**, un réseau de neurones développé par Google pour la reconnaissance de sons génériques, préentraîné sur le dataset **AudioSet**.

1. **Extraction des embeddings audio**  
   Chaque fichier `.wav` est chargé et rééchantillonné à **16 kHz** pour correspondre aux exigences de YAMNet.  
   Le modèle YAMNet extrait un **vecteur d’embedding** pour chaque segment audio, et la moyenne de ces vecteurs constitue la représentation globale du fichier.  

2. **Classification supervisée**  
   Un modèle personnalisé, `yamnet_custom_classifier.h5`, est ensuite chargé pour effectuer la classification finale.  
   Ce modèle a été entraîné sur les embeddings YAMNet issus de l’ensemble d’entraînement, associés aux catégories du corpus MSR-VTT.  
   Les prédictions sont comparées aux vraies étiquettes via un **rapport de classification** (précision, rappel, F1-score) et une **matrice de confusion** visualisée avec *Seaborn*.

3. **Fichiers et organisation**  
   - Les catégories sont définies dans `train_val_annotation/category.txt`  
   - Les annotations de test sont lues depuis `test_videodatainfo.json/test_videodatainfo.json`  
   - Les fichiers audio correspondants sont recherchés dans le répertoire `test_val_audios/`

##### c. Résultats et remarques

Le modèle YAMNet couplé au classifieur personnalisé permet d’obtenir une **bonne discrimination entre les 20 classes** audio-visuelles du corpus, notamment pour les catégories à signature sonore forte (musique, sport, discours, etc.).  
Une matrice de confusion est produite pour analyser les confusions entre classes et orienter les ajustements futurs du modèle.


#### 3. Texte

## Méthodologie

### Pipeline de traitement

Le projet consiste à classifier des captions de vidéos en différentes catégories. La méthodologie suivie se déroule en plusieurs étapes :

1. **Prétraitement des données**  
   - Les captions sont tokenisées à l'aide des tokenizers de Transformers (DistilBERT ou ALBERT selon le modèle).  
   - Les séquences sont tronquées ou paddées pour obtenir une longueur uniforme (`max_len=128`).  
   - Optionnellement, les captions peuvent être **agrégées par vidéo**, en concaténant toutes les captions d’une même vidéo pour créer un contexte plus riche.

2. **Jeux de données**  
   - Les données sont divisées en ensembles `train`, `validation` et `test`.  
   - Les labels sont convertis en indices numériques pour le training.  
   - Des DataLoaders PyTorch permettent un traitement par batch efficace.

3. **Modèles testés**  
   Nous avons expérimenté avec **3 modèles principaux** :

   - **Modèle 1 : LSTM avec DistilBERT gelé**  
     DistilBERT est utilisé comme encodeur, mais ses poids sont gelés. La sortie est passée dans un LSTM profond à 4 couches suivi de plusieurs couches linéaires avec ReLU et Dropout.

   - **Modèle 2 : ALBERT avec AttentionPooling (meilleur modèle)**  
     ALBERT est entièrement fine-tuné.  
     Chaque token est pondéré via un module **AttentionPooling**, qui calcule un score d’attention et crée un vecteur de contexte moyen.  
     On concatène ce vecteur avec l’embedding avant de passer dans le classifieur final.  
     Ce modèle supporte également l’**agrégation de toutes les captions d’une vidéo**, ce qui améliore la qualité de la représentation et la précision.  
     Ce modèle a obtenu les meilleures performances en terme d’accuracy.

   - **Modèle 3 : DistilBERT simple fine-tuné**  
     DistilBERT est entièrement fine-tuné avec un classifieur simple à quelques couches linéaires. Rapide à entraîner et efficace sur de petits datasets.

4. **Entraînement et évaluation**  
   - Optimisation avec **AdamW** et scheduler de learning rate linéaire avec warmup.  
   - Suivi de la **loss** et de l’**accuracy** à chaque epoch pour l’entraînement et la validation.  
   - Évaluation finale sur le jeu de test avec affichage de la **matrice de confusion normalisée** et du rapport de classification.

5. **Résultats et conclusion**  
   - Le modèle ALBERT avec AttentionPooling et aggregation des captions est le plus performant.  
   - L’utilisation de l’attention permet de mettre en valeur les tokens les plus importants, et l’agrégation apporte un contexte global pour chaque vidéo.  
   - Une comparaison visuelle des performances des trois modèles est disponible ci-dessous :

<p align="center">
  <img src="/texte/comparaison_modele.png" alt="Comparaison des modèles" width="800">
</p>



### Fusion pour la classification multimodale

## Conclusion
