# 🦉 Birdwatcher – Système autonome d'observation et d'enregistrement vidéo en pleine nature

Birdwatcher est une application **Flask** pour Raspberry Pi, conçue pour l’observation de la faune (et particulièrement les oiseaux) en conditions de terrain.  
Elle combine **détection de mouvements** et **IA (YOLOv8)** pour déclencher l’enregistrement vidéo, avec une interface web locale optimisée pour mobile et utilisable même sans connexion Internet.

---

## Fonctionnalités

- **Interface web immersive** (Flask + TailwindCSS + DaisyUI en local)
- **Mode offline** : tous les assets CSS/JS sont hébergés localement
- **Détection PIR** pour déclencher l’enregistrement
- **Détection IA (YOLOv8)** pour filtrer les événements pertinents
- **Gestion des vidéos** : lecture, suppression, téléchargement
- **Compatibilité mobile** avec navigation fixe et plein écran
- **Optimisé pour Raspberry Pi 4 en déploiement terrain**

---

## Architecture

Raspberry Pi 4 ├── app.py # Serveur Flask 
├── detect_capture.py # Détection PIR + capture vidéo 
├── yolov8n.pt # Modèle IA YOLOv8n 
├── templates/ # Pages HTML 
├── static/css/ # Styles Tailwind + DaisyUI locaux 
├── videos/ # Stockage des vidéos 
└── logs/ # Journaux système

---

## Installation

### 1. Prérequis
- Raspberry Pi 4 (ou modèle compatible)
- OS : Raspberry Pi OS
- Python 3.13
- Caméra compatible
- Capteur PIR (ex. HC‑SR501)
- Accès SSH ou écran/clavier

### 2. Cloner le dépôt
git clone https://github.com/babaygo/Birdwatcher.git
cd Birdwatcher
git checkout dev

### 3. Installer les dépendances Python
pip install -r requirements.txt

### 4. Lancer l'application
python app.py

## Utilisation
Connectez-vous au wi-fi local puis, ouvrez un navigateur et accédez à :
http://birdwatcher.local (ou : http://<ip_de_la_pi>:5000)

L’interface permet de :
- Visualiser les vidéos
- Consulter les métadonnées des vidéos (indice de confiance de la détecttion, timestamp)
- Supprimer ou télécharger les vidéos
