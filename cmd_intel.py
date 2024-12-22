import speech_recognition as sr
import pyttsx3
import subprocess
import cv2
import dlib
import numpy as np
import pickle
from PIL import Image
import pytesseract
from gtts import gTTS
from playsound import playsound
import os
from ultralytics import YOLO
import time

# Spécifiez le chemin vers le binaire de Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' 


# Initialiser le moteur de synthèse vocale
engine = pyttsx3.init()

def parler(texte):
    """Fait parler l'ordinateur."""
    engine.say(texte)
    engine.runAndWait()

def ecouter_commande():
    """Écoute et retourne la commande vocale."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Je vous écoute...")
        try:
            # Ajuster le bruit ambiant
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source, timeout=5)
            commande = recognizer.recognize_google(audio, language='fr-FR')
            print(f"Vous avez dit : {commande}")
            return commande.lower()
        except sr.UnknownValueError:
            parler("Je n'ai pas compris, pouvez-vous répéter ?")
        except sr.RequestError:
            parler("Il y a un problème avec le service de reconnaissance vocale.")
        except sr.WaitTimeoutError:
            parler("Temps écoulé, veuillez réessayer.")
        return None

def detection_objets():
    """Lance la détection d'objets en temps réel avec YOLO et décrit les objets détectés une seule fois."""
    try:
        # Charger le modèle YOLO
        model = YOLO("yolov8n.pt")  # Chemin vers votre modèle YOLO personnalisé

        # Ouvrir la caméra
        cap = cv2.VideoCapture(0)

        # Vérifier si la caméra s'ouvre correctement
        if not cap.isOpened():
            parler("Erreur : Impossible d'ouvrir la caméra.")
            return

        # Définir la résolution de la caméra pour améliorer la performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Largeur
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Hauteur

        parler("Détection d'objets en cours. Appuyez sur la touche 'q' pour arrêter.")

        # Variables pour contrôler l'annonce unique des objets
        objets_detectes_precedemment = set()
        temps_derniere_annonciation = time.time()
        intervalle_annonciation = 2  # Annonce toutes les 2 secondes maximum

        while True:
            # Lire une image de la caméra
            ret, frame = cap.read()
            if not ret:
                parler("Erreur : Impossible de lire depuis la caméra.")
                break

            # Faire des prédictions avec le modèle YOLO
            results = model(frame, conf=0.25)

            # Extraire les étiquettes des objets détectés
            objets_detectes = results[0].names
            detections = results[0].boxes

            # Créer une liste pour les objets détectés
            description_objets = []

            for detection in detections:
                label = int(detection.cls[0])  # Obtenir l'étiquette de l'objet
                objet = objets_detectes[label]  # Trouver le nom de l'objet
                description_objets.append(objet)

            # Si des objets sont détectés, les annoncer une seule fois
            objets_actuels = set(description_objets)  # Étiquettes des objets détectés dans ce frame
            nouveaux_objets = objets_actuels - objets_detectes_precedemment  # Objets nouveaux depuis la dernière image

            if nouveaux_objets:
                if time.time() - temps_derniere_annonciation > intervalle_annonciation:
                    # Annonce des nouveaux objets détectés
                    parler("Objet détecté: " + ", ".join(nouveaux_objets))
                    objets_detectes_precedemment.update(nouveaux_objets)  # Mettre à jour les objets déjà détectés
                    temps_derniere_annonciation = time.time()  # Réinitialiser le temps de la dernière annonce

            # Annoter l'image avec les résultats
            annotated_frame = results[0].plot()

            # Afficher les résultats en temps réel
            cv2.imshow("Détection d'objets", annotated_frame)

            # Quitter avec la touche 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                parler("Détection d'objets arrêtée.")
                break

        # Libérer les ressources
        cap.release()
        cv2.destroyAllWindows()

    except Exception as e:
        parler(f"Une erreur est survenue : {e}")

def reconnaissance_faciale():
    """Lance la reconnaissance faciale en temps réel et annonce les noms détectés une seule fois."""
    parler("Lancement de la reconnaissance faciale.")
    try:
        # Charger les modèles Dlib
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('C:/Users/pc/Desktop/face/shape_predictor_68_face_landmarks.dat')
        face_encoder = dlib.face_recognition_model_v1('C:/Users/pc/Desktop/face/dlib_face_recognition_resnet_model_v1.dat')

        # Charger ou initialiser la base de données des visages connus
        try:
            with open('known_faces.pkl', 'rb') as f:
                known_face_descriptors, known_face_names = pickle.load(f)
        except FileNotFoundError:
            known_face_descriptors, known_face_names = [], []

        def get_face_descriptor(image, face):
            """Extrait un descripteur de visage."""
            shape = predictor(image, face)
            return np.array(face_encoder.compute_face_descriptor(image, shape))

        # Variables pour contrôler les annonces uniques des noms
        personnes_detectees_precedemment = set()
        temps_derniere_annonciation = time.time()
        intervalle_annonciation = 5  # Annonce toutes les 5 secondes maximum

        # Reconnaissance faciale en temps réel
        cap = cv2.VideoCapture(0)
        parler("Appuyez sur la touche 'q' pour arrêter.")

        while True:
            ret, frame = cap.read()
            if not ret:
                parler("Erreur : Impossible de lire depuis la caméra.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            noms_detectes = set()

            for face in faces:
                try:
                    # Obtenir le descripteur du visage
                    face_descriptor = get_face_descriptor(frame, face)
                    distances = [np.linalg.norm(face_descriptor - known) for known in known_face_descriptors]
                    min_distance = min(distances) if distances else None

                    # Déterminer le nom associé au visage
                    if min_distance is not None and min_distance < 0.6:
                        match_index = distances.index(min_distance)
                        name = known_face_names[match_index]
                    else:
                        name = "Inconnu"

                    noms_detectes.add(name)

                    # Dessiner une boîte et ajouter le nom détecté sur l'image
                    cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
                    cv2.putText(frame, name, (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                except Exception as e:
                    print(f"Erreur lors du traitement d'un visage : {e}")

            # Identifier les nouvelles personnes détectées depuis le dernier frame
            nouvelles_personnes = noms_detectes - personnes_detectees_precedemment

            if nouvelles_personnes:
                if time.time() - temps_derniere_annonciation > intervalle_annonciation:
                    # Annoncer les nouvelles personnes détectées
                    parler("Personnes détectées: " + ", ".join(nouvelles_personnes))
                    personnes_detectees_precedemment.update(nouvelles_personnes)
                    temps_derniere_annonciation = time.time()

            # Afficher la vidéo annotée
            cv2.imshow("Reconnaissance Faciale", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                parler("Arrêt de la reconnaissance faciale.")
                break

        cap.release()
        cv2.destroyAllWindows()

    except Exception as e:
        parler("Une erreur s'est produite lors de la reconnaissance faciale.")
        print(f"Erreur : {e}")

def texte_en_audio():
    """Convertit du texte extrait d'une image en audio et le joue."""
    try:
        # Chemin de l'image à analyser
        image_path = "C:\\Users\\pc\\Pictures\\test3.jpg"  # Changez le chemin si nécessaire
        
        # Charger l'image et extraire le texte
        print("Chargement de l'image et extraction du texte...")
        image = Image.open(image_path)
        texte_extrait = pytesseract.image_to_string(image, lang='fra')  # 'fra' pour le français
        
        if not texte_extrait.strip():
            print("Aucun texte détecté dans l'image.")
            return

        print("Texte extrait de l'image :\n", texte_extrait)
        
        # Conversion du texte en audio
        print("Conversion du texte en audio...")
        tts = gTTS(text=texte_extrait, lang='fr', slow=False)
        
        # Sauvegarder et lire l'audio
        audio_path = "audio.mp3"
        tts.save(audio_path)
        print("Lecture de l'audio...")
        playsound(audio_path)

        # Nettoyage après lecture
        os.remove(audio_path)
        print("Fichier audio supprimé après lecture.")
    except FileNotFoundError:
        print("Le fichier image spécifié est introuvable.")
    except pytesseract.TesseractError as e:
        print(f"Erreur avec Tesseract : {e}")
    except Exception as e:
        print(f"Une erreur s'est produite : {e}")

def detection_money():
    """Lance la détection de la monnaie marocaine en temps réel avec un modèle YOLO personnalisé et annonce les valeurs détectées."""
    try:
        # Charger le modèle personnalisé pour la détection de la monnaie marocaine
        model = YOLO("C:/Users/pc/Desktop/face/bestm.pt")  # Remplacez par le chemin vers votre modèle YOLO personnalisé

        # Ouvrir la caméra
        cap = cv2.VideoCapture(0)

        # Vérifier si la caméra s'ouvre correctement
        if not cap.isOpened():
            parler("Erreur : Impossible d'ouvrir la caméra.")
            return

        # Définir la résolution de la caméra pour améliorer la performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Largeur
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Hauteur

        parler("Détection de la monnaie marocaine en cours. Appuyez sur la touche 'q' pour arrêter.")

        # Variables pour contrôler l'annonce unique des monnaies
        monnaies_detectees_precedemment = set()
        temps_derniere_annonciation = time.time()
        intervalle_annonciation = 2  # Annonce toutes les 2 secondes maximum

        while True:
            # Lire une image de la caméra
            ret, frame = cap.read()
            if not ret:
                parler("Erreur : Impossible de lire depuis la caméra.")
                break

            # Faire des prédictions avec le modèle
            results = model(frame, conf=0.25)

            # Extraire les étiquettes des monnaies détectées
            monnaies_detectees = results[0].names
            detections = results[0].boxes

            # Créer une liste pour les monnaies détectées
            description_monnaies = []

            for detection in detections:
                label = int(detection.cls[0])  # Obtenir l'étiquette de la monnaie
                monnaie = monnaies_detectees[label]  # Trouver le nom ou la valeur de la monnaie
                description_monnaies.append(monnaie)

            # Si des monnaies sont détectées, les annoncer une seule fois
            monnaies_actuelles = set(description_monnaies)  # Étiquettes des monnaies détectées dans ce frame
            nouvelles_monnaies = monnaies_actuelles - monnaies_detectees_precedemment  # Nouvelles monnaies détectées

            if nouvelles_monnaies:
                if time.time() - temps_derniere_annonciation > intervalle_annonciation:
                    # Annonce des nouvelles monnaies détectées
                    parler("Monnaie détectée : " + ", ".join(nouvelles_monnaies))
                    monnaies_detectees_precedemment.update(nouvelles_monnaies)  # Mettre à jour les monnaies déjà détectées
                    temps_derniere_annonciation = time.time()  # Réinitialiser le temps de la dernière annonce

            # Annoter l'image avec les résultats
            annotated_frame = results[0].plot()

            # Afficher les résultats en temps réel
            cv2.imshow("Détection de monnaie marocaine", annotated_frame)

            # Quitter avec la touche 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                parler("Détection de monnaie arrêtée.")
                break

        # Libérer les ressources
        cap.release()
        cv2.destroyAllWindows()

    except Exception as e:
        parler(f"Une erreur est survenue : {e}")

# Dictionnaire des commandes
commandes = {
    "détection d'objet": detection_objets,
    "reconnaissance faciale": reconnaissance_faciale,
    "audio": texte_en_audio,
     "folos": detection_money,
}

# Programme principal
if __name__ == "__main__":
    parler("Dites 'mes lunettes' pour activer vos lunettes intelligentes.")
    while True:
        print("En attente de l'activation avec 'mes lunettes'...")
        activation = ecouter_commande()
        if activation and "mes lunettes" in activation:
            parler("Je suis vos lunettes intelligentes. je vous écoute.")
            while True:
                commande = ecouter_commande()
                if commande:
                    if "quitter" in commande:
                        parler("Arrêt du programme. À bientôt!")
                        exit()
                    elif commande in commandes:
                        commandes[commande]()
                    else:
                        parler("Commande non reconnue. Veuillez réessayer.")
