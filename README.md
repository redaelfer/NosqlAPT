# NosqlAPT - Détection d'APT via Analyse de Logs NoSQL 🛡️

**NosqlAPT** est un système de détection de menaces persistantes avancées (APT) conçu pour surveiller les environnements NoSQL. Le projet utilise une architecture hybride combinant l'apprentissage automatique (XGBoost) et l'apprentissage profond (Autoencoders) pour identifier des comportements malveillants à travers les journaux système.

## 🌟 Fonctionnalités Clés

* **Analyse de Logs Multi-niveaux** : Traitement et normalisation des logs système pour l'extraction de caractéristiques pertinentes.
* **Détection Hybride IA** :
* **Autoencoder** : Utilisé pour la détection d'anomalies non supervisée.
* **XGBoost** : Utilisé pour la classification supervisée des étapes d'une attaque APT.


* **Pipeline de Prétraitement** : Nettoyage et encodage automatique des données via un pipeline Scikit-learn sérialisé.
* **Architecture Conteneurisée** : Déploiement simplifié de la stack ELK (Logstash) et des services de détection via Docker Compose.

## 🛠️ Technologies Utilisées

* **Langage** : Python 3.9.
* **Intelligence Artificielle** : TensorFlow/Keras (Autoencoders), XGBoost, Scikit-learn.
* **Infrastructure** : Docker, Docker Compose, Logstash.
* **Data Science** : Pandas, NumPy, Joblib.

## Installation et Lancement

### 1. Prérequis

* Docker et Docker Compose installés.
* Python 3.9 (pour l'exécution locale des scripts).

### 2. Déploiement via Docker

Le projet utilise Docker Compose pour orchestrer les services de collecte et de traitement :

```bash
# Lancement de la stack (Logstash et services associés)
docker-compose up --build

```

### 3. Utilisation des Scripts de Détection

Vous pouvez tester le détecteur avec les scripts fournis dans le dossier `scripts/` ou à la racine :

```bash
# Lancer la démo de détection
python scripts/apt_detection_demo.py

# Intégrer et traiter les logs
python process_logs.py

```

## 📂 Structure du Projet

* `apt_detection_project/models/` : Contient les modèles entraînés (`.h5`, `.json`) et le préprocesseur (`.pkl`).
* `apt_detection_project/src/` : Coeur de la logique de détection et de préparation des données.
* `logstash/config/` : Configuration de l'ingestion des logs via Logstash.
* `apt_detector.py` : Script principal pour l'interface de détection.

## 📊 Pipeline de Détection

1. **Ingestion** : Les logs sont collectés et envoyés vers le pipeline de traitement.
2. **Prétraitement** : Les données sont nettoyées et transformées selon les caractéristiques attendues (`expected_features.json`).
3. **Analyse** : Le modèle hybride évalue si le comportement correspond à une étape d'attaque APT.
4. **Alerte** : Les résultats sont consignés dans les fichiers de logs de détection.

---
