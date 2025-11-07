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

#### 3. Texte

### Fusion pour la classification multimodale

## Conclusion
