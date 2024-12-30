Introduction
============
Projet :
- Lunettes intelligentes pour malvoyantes

Réalisé par :
- lemlih Sabah
- El Achabi Aya

Encadré par :
- M.Masrour

But :

Offrir une solution innovante permettant aux personnes malvoyantes de mieux comprendre leur environnement grâce à des descriptions vocales générées en temps réel.

Fonctionnalités principales :

- Une caméra intégrée pour capturer l’environnement de l’utilisateur.
- Un système d’intelligence artificielle capable de détecter et de reconnaître des objets en temps réel.
- Un haut-parleur intégré pour fournir des descriptions vocales des objets détectés.
- Une interface vocale permettant à l’utilisateur de commander le dispositif de manière intuitive.

echnologies utilisées :

- Vision par ordinateur et réseaux neuronaux pour la reconnaissance d’objets.
- Modèles d’apprentissage profond pour une détection précise et rapide.

Utilisation
===========
Fonctionnalités principales
1. Reconnaissance vocale
Les lunettes écoutent les commandes vocales de l'utilisateur grâce à la bibliothèque speech_recognition. Une fois activées avec le mot-clé "mes lunettes", elles exécutent l'une des commandes suivantes :

"détection d'objet" : Lance la détection d'objets en temps réel.

"reconnaissance faciale" : Identifie les visages connus et annonce leurs noms.

"audio" : Lit un texte à partir d'une image et le convertit en audio.

"folos" : Effectue la détection des monnaies marocaines.

2. Détection d'objets
Utilisant le modèle YOLOv8, les lunettes détectent et annoncent les objets présents dans l'environnement de l'utilisateur. La détection s'effectue en temps réel via la caméra embarquée. Chaque objet détecté est annoncé une seule fois, évitant des répétitions inutiles.

3. Reconnaissance faciale
La reconnaissance faciale est réalisée à l'aide des modèles de dlib pour identifier les visages connus. Les visages sont comparés avec une base de données préenregistrée, et les noms correspondants sont annoncés à l'utilisateur.

4. Conversion de texte en audio
Les lunettes capturent une image contenant du texte, extraient le contenu texte à l'aide de Tesseract OCR, puis le transforment en audio grâce à la bibliothèque gTTS. Le fichier audio est lu directement après la conversion.

5. Détection de monnaies marocaines
Grâce à un modèle YOLO personnalisé, les lunettes détectent différentes dénominations de la monnaie marocaine et annoncent leur valeur à l'utilisateur.



### Pré-requis

Pour utiliser ce projet, vous devez avoir :
- Technologies utilisées
- Python : Langage principal pour le développement.
- YOLOv8 : Modèle de vision par ordinateur pour la détection d'objets et de monnaies.
- dlib : Bibliothèque utilisée pour la reconnaissance faciale.
- Tesseract OCR : Extraction de texte à partir d'images.
- speech_recognition : Pour la reconnaissance des commandes vocales.
- gTTS et playsound : Pour la synthèse vocale et la lecture des fichiers audio.
- OpenCV : Gestion de la caméra et affichage des résultats en temps réel.



### schéma 

smart-glasses-project/

│

├── src/

│   ├── face_recognition.py     # Module de reconnaissance faciale

│   ├── object_detection.py     # Module pour détecter les objets environnants

│   ├── money_detection.py     # Module pour détecter la monnaie 

│   ├── voice_command.py        # Module de gestion des commandes vocales

│   ├── image_to_text.py        # Module pour convertir l'image en texte

│   └── text_to_audio.py        # Module pour convertir le texte en audio

│

├── models/                     # Modèles pré-entraînés et données

│   ├── yolov8.pt          # Modèle YOLO pour détection d'objets

│   ├── best.pt            # Modèle YOLO pour détection de money

│   ├── embeddings/             # Embeddings des visages

│   │   └── embeddings_db.npy

│   └── tesseract/              # Modèles OCR pour la reconnaissance de texte

│
