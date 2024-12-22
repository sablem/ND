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