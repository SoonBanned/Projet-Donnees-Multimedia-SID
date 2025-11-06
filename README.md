# Projet Master 2 SID

## *Classification et Recherche Cross-Modale sur la collection MSR-VTT*

## Introduction

Ce projet s’inscrit dans le cadre de notre projet de Master 2 qui porte sur la classificationde vidéos à partir de la collection **MSR-VTT**, un jeu de données de référence en traitement de données multimédia.

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

#### 2. Audio

#### 3. Texte

### Fusion pour la classification multimodale

## Conclusion