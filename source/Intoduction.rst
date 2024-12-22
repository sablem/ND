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
============
### Pré-requis

Pour utiliser ce projet, vous devez avoir :
- Python 3.7 ou plus récent.
- pip installé.
- Bibliothèques Python : OpenCV, PyTorch/TensorFlow, gTTS ...


### schema 

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