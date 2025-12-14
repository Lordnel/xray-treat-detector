# Détection d'objets dangereux dans les bagages à rayons X avec YOLOv11m – Pipeline DevOps / MLOps

Projet tutoré : mise en place d’un mini pipeline **DevOps–MLOps** pour la détection d’objets dangereux (armes, couteaux, outils…) dans des images de bagages scannés aux rayons X.

Le projet combine :

- **MLOps** : entraînement, évaluation et sauvegarde d’un modèle YOLOv11m fine-tuné sur le dataset **SIXray** (version Roboflow),
- **DevOps** : API Flask, conteneurisation Docker, pipeline CI GitHub Actions, déploiement sur **AWS EC2**.

---

## 1️⃣ Contexte & objectifs

Dans les systèmes de sécurité aéroportuaire, la détection d’armes dans les bagages est une tâche critique.  
Aujourd’hui, l’analyse des images rayons X repose largement sur des opérateurs humains, ce qui peut entraîner :

- de la fatigue,
- des erreurs de détection,
- des temps de traitement importants.

**Objectif du projet :**

> Entraîner un modèle de vision par ordinateur (YOLOv11m) pour détecter automatiquement des objets dangereux dans des images rayons X, puis l’exposer via une API conteneurisée et déployée dans le cloud (AWS EC2), avec un pipeline CI/CD.

---

## 2️⃣ Dataset

- **Nom** : SIXray (version Roboflow)
- **Type** : images rayons X de bagages avec annotations bounding boxes
- **Tâche** : détection d’objets (armes et outils)
- **Format** : YOLO (train / valid / test + `data.yaml`)

Téléchargement via Roboflow dans le notebook :

```python
from roboflow import Roboflow

rf = Roboflow(api_key="VOTRE_API_KEY")
project = rf.workspace("siewchinyip-outlook-my").project("sixray")
dataset = project.version(4).download("yolov8")  # compatible YOLOv11
```

Les scripts d’entraînement, d’EDA et d’évaluation se trouvent dans :

```text
notebooks/Sixray_YOLOv11m_Training.ipynb
```

---

## 3️⃣ Architecture globale (DevOps / MLOps)

Le pipeline global se décompose en plusieurs étapes.

### 3.1 MLOps – Entraînement du modèle

1. Préparation des données (Roboflow → format YOLO).
2. Analyse exploratoire (distribution des classes, exemples d’images, complexité des données).
3. Entraînement de YOLOv11m (fine-tuning) sur AWS (SageMaker / Notebook).
4. Visualisation des courbes d’apprentissage (loss, mAP50, mAP50-95).
5. Évaluation du modèle (mAP, précision, rappel, matrice de confusion).
6. Sauvegarde du meilleur modèle : `model/yolo11m_sixray_best.pt`.

### 3.2 Backend – API Flask

L’API expose deux endpoints principaux :

- `GET /health` : vérifie que le service est prêt et que le modèle est chargé.
- `POST /predict` : prend en entrée une image rayons X (multipart/form-data) et renvoie les objets détectés (classes, score, bounding boxes).

Fichier principal : `api/app.py`.

### 3.3 DevOps – Conteneurisation Docker

- Dockerfile dans `docker/Dockerfile`
- Image basée sur `python:3.10-slim`
- Installation de Flask, Ultralytics (YOLO), OpenCV, NumPy
- Copie du modèle et de l’API dans l’image
- Exposition du port 5000 pour l’API

### 3.4 CI – GitHub Actions

- Fichier `.github/workflows/ci.yml`
- Déclenchement automatique sur `push` / `pull_request` sur `main`
- Étapes :
  - Récupération du code
  - Installation de Python et des dépendances
  - Exécution des tests unitaires (`pytest`)
  - (Optionnel) Build de l’image Docker pour vérifier que le Dockerfile est valide

### 3.5 CD – Déploiement sur AWS EC2

- Instance EC2 (Ubuntu) avec Docker installé
- Script `deploy.sh` présent sur l’EC2 pour :
  - récupérer la dernière version du code (`git pull`),
  - construire l’image Docker,
  - arrêter l’ancien conteneur,
  - lancer la nouvelle version de l’API.
- Un job **`deploy`** dans GitHub Actions se connecte en **SSH** à l’EC2 et exécute `deploy.sh` après la réussite des tests et du build Docker.

Ainsi, le pipeline complète la boucle **CI/CD** de manière entièrement automatisée :  
**push sur GitHub → tests → build → déploiement sur EC2**.

Un schéma d’architecture détaillé peut être placé dans `docs/architecture_devops_mlops.png`.

---

## 4️⃣ Structure du projet

```text
xray-yolov11-security/
├── api/                       # API Flask (inférence)
│   ├── app.py
├── models/                    # Modèle YOLOv11m fine-tuné
│   └── yolo11m_sixray_best.pt
├── notebooks/                 # Entraînement & EDA
│   └── Sixray_YOLOv11m_Training.ipynb
├── docker/                    # Conteneurisation
│   └── Dockerfile
├── tests/                     # Tests unitaires (CI)
│   └── test_health.py
├── docs/                      # Ressources pour le rapport
│   ├── architecture_devops_mlops.png
│   ├── results_curves.png
│   ├── confusion_matrix.png
│   └── samples_predictions/
│       ├── val_batch0_pred.jpg
│       ├── val_batch1_pred.jpg
│       └── ...
├── .github/
│   └── workflows/
│       └── ci.yml
├── deploy.sh
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 5️⃣ Installation & exécution locale

### 5.1 Prérequis

- Python 3.10+
- `pip`
- (Optionnel) Docker

### 5.2 Installation (sans Docker)

```bash
git clone https://github.com/<votre-user>/xray-yolov11-security.git
cd xray-yolov11-security

# Installation des dépendances
pip install -r requirements.txt
```

Assurez-vous que le modèle existe :

```text
model/yolo11m_sixray_best.pt
```

(soit versionné dans le repo, soit copié depuis votre environnement d'entraînement).

### 5.3 Lancement de l’API Flask

```bash
python api/app.py
```

L’API est accessible sur :  
`http://localhost:5000`

- **Health check :**

```bash
curl http://localhost:5000/health
```

- **Prédiction sur une image :**

```bash
curl -X POST http://localhost:5000/predict   -F "image=@path/to/xray_image.jpg"
```

---

## 6️⃣ Utilisation avec Docker

### 6.1 Build de l’image

```bash
docker build -t xray-yolo -f docker/Dockerfile .
```

### 6.2 Lancer le conteneur

```bash
docker run -d --name xray-api -p 5000:5000 xray-yolo
```

Puis :

```bash
curl http://localhost:5000/health
```

---

## 7️⃣ CI – Intégration Continue (GitHub Actions)

Le pipeline CI se trouve dans :

```text
.github/workflows/ci.yml
```

Il réalise :

1. **Checkout** du code,
2. Installation de Python et des dépendances,
3. Lancement des tests unitaires :

   - Fichier de test principal : `tests/test_health.py`
   - Vérifie que l’API répond correctement sur `/health`.

4. (Optionnel) Vérification du build Docker.


---

## 8️⃣ CD – Déploiement automatisé sur AWS EC2 via GitHub Actions

Le **déploiement continu (CD)** est également géré par GitHub Actions, grâce à un job `deploy` qui :

1. s’exécute uniquement sur la branche `main`,
2. attend que les jobs `test` et `docker-build` soient passés avec succès,
3. se connecte en **SSH** à l’instance EC2,
4. exécute le script `deploy.sh` sur la machine distante.

Ainsi, à chaque **push sur `main`** :

1. les tests sont exécutés,
2. l’image Docker est construite,
3. si tout est OK, le code est déployé automatiquement sur l’EC2.

L’API est alors accessible via :  
`http://<IP_EC2>:5000/health`

---

## 9️⃣ Limites & perspectives

### 9.1 Limites actuelles

- Le déploiement actuel entraîne une courte indisponibilité lors de la mise à jour du conteneur (l’ancien est stoppé avant le redémarrage du nouveau).
- Le modèle est statique : il n’y a pas encore de mécanisme de re-entraînement automatique ou de gestion de versions de modèles.

### 9.2 Pistes d’amélioration

- Implémenter un déploiement **Blue/Green** pour garantir un déploiement sans interruption de service (zéro downtime).
- Ajouter un système de **monitoring** (logs, métriques, alertes) pour suivre en temps réel :
  - les performances de l’API (latence, erreurs),
  - les performances du modèle (taux de détection, dérive des données).
- Intégrer un outil de gestion des expériences et des modèles (MLflow, DVC) pour :
  - versionner les jeux de données,
  - tracer les expériences d’entraînement,
  - gérer plusieurs modèles en parallèle.
- Déployer l’API derrière un **reverse proxy** (Nginx) ou un **load balancer** pour faciliter le scaling horizontal.

---

## Résumé

Ce projet illustre un cas d’usage complet de **vision par ordinateur** appliquée à la **sécurité aéroportuaire**, depuis :

- l’entraînement d’un modèle YOLOv11m sur un dataset complexe (SIXray),
- jusqu’à son déploiement dans un pipeline **DevOps–MLOps** reproductible sur **AWS EC2**.

Il peut servir de base à des extensions plus avancées (monitoring, gestion des versions de modèles, détection temps réel, amélioration de la robustesse du déploiement, etc.).
