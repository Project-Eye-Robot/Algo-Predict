# Eye-Robot : Algo-Predict
## OpenCV
Solution 1: Choix de l'open source: **OpenCV**

### What is Open CV ?
OpenCV was designed for computational efficiency and with a strong focus on real-time applications. Written in optimized C/C++, the library can take advantage of multi-core processing. Enabled with OpenCL, it can take advantage of the hardware acceleration of the underlying heterogeneous compute platform.

- https://github.com/opencv/opencv

### Doc technique

#### Détection d'objets
https://github.com/opencv/opencv/tree/master/data/haarcascades --> code pour détection d’objet
- https://www.framboise314.fr/i-a-realisez-un-systeme-de-reconnaissance-dobjets-avec-raspberry-pi/

### Doc d'installation
- https://www.framboise314.fr/i-a-realisez-un-systeme-de-reconnaissance-dobjets-avec-raspberry-pi/

### Software Installation

#### Installation de OpenCV avec pip
- $ sudo apt-get install libhdf5-dev libhdf5-serial-dev
- $ sudo apt-get install libqtwebkit4 libqt4-test

#### 1.1) Installez pip avec ces 2 commandes,

- $ wget https://bootstrap.pypa.io/get-pip.py
- $ sudo python3 get-pip.py

#### 1.2) ainsi qu’ OpenCV et le module Python de la PI caméra.

- $ sudo pip install opencv-contrib-python
- $ sudo pip install "picamera[array]"
- $ sudo pip install imutils
- $ sudo pip install pyautogui

#### 1.3) Installez pip avec ces 2 commandes,

$ sudo apt-get install libatlas-base-dev
$ sudo apt-get install libjasper-dev
$ sudo apt-get install libqtgui4
$ sudo apt-get install python3-pyqt5

#### Tests de bon fonctionnement

#### 2.1) Ouvrez la console de Python avec cette commande :

$ python3

#### 2.2) Pour vérifier qu’OpenCV est opérationnel, tapez cette commande :

>>> import cv2

Si vous n’avez pas de message d’erreur, OpenCV est bien installé 
#### 2.3) Vous pouvez connaître la version en entrant cette commande :

>>> cv2.__version__

#### 2.4) Pour sortir et “tuer” le processus de l’interpréteur Python, tapez Ctrl + d.
 
Avant de rentrer dans le “grand bain” de la vision artificielle, testons la Pi Caméra, avec un  programme Python, qui ouvre une fenêtre et une vidéo en direct.
 
#### 2.5) Créez dans le répertoire de votre choix, un fichier picamera.py et écrivez le code suivant : 

# test de la caméra picamera.py
# importer les paquets requis pour la Picaméra
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
 
# initialisation des paramètres pour la capture
camera = PiCamera()
camera.resolution = (800, 600)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(800, 600))
 
# temps réservé pour l'autofocus
time.sleep(0.1)
 
# capture du flux vidéo
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
 
# recupère à l'aide de Numpy le cadre de l'image, pour l'afficher ensuite à l'écran
             image = frame.array
 
# affichage du flux vidéo
             key = cv2.waitKey(1) & 0xFF
 
# initialisation du flux 
             rawCapture.truncate(0)
 
# si la touche q du clavier est appuyée, on sort de la boucle
             if key == ord("q"):
                        break

#### 2.6) Pour afficher la vidéo en streaming, tapez cette commande dans le terminal :

$ python3 picamera.py

#### Le programme Python

#### 3.1) Commencez par créer un répertoire, nommons le “reconnaissance_objets” afin de stocker les fichiers nécessaires :
             
- le programme Python : reconnaissance_objets.py
- Le fichier entraîné aux 21 types d’objets : MobileNetSSD_deploy.caffemodel
- Le fichier de configuration : MobileNetSSD_deploy.prototxt

Ci dessous, le code du programme, reconnaissance_objets.py :

# Ouvrir un terminal et executer la commande ci dessous
# python3 reconnaissance_objets.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# importer tout les packages requis
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
# packages nécessaires pour la gestion des emails
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import ntpath
import pyautogui


# construction des arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
    help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
    help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialiser la liste des objets entrainés par MobileNet SSD 
# création du contour de détection avec une couleur attribuée au hasard pour chaque objet
CLASSES = ["arriere-plan", "avion", "velo", "oiseau", "bateau",
    "bouteille", "autobus", "voiture", "chat", "chaise", "vache", "table",
    "chien", "cheval", "moto", "personne", "plante en pot", "mouton",
    "sofa", "train", "moniteur"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# chargement des fichiers depuis le répertoire de stockage 
print(" ...chargement du modèle...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialiser la caméra du pi, attendre 2s pour la mise au point ,
# initialiser le compteur FPS
print("...démarrage de la Picamera...")
vs = VideoStream(usePiCamera=True, resolution=(1600, 1200)).start()
time.sleep(2.0)
fps = FPS().start()

# boucle principale du flux vidéo
while True:
    # récupération du flux vidéo, redimension 
    # afin d'afficher au maximum 800 pixels 
    frame = vs.read()
    frame = imutils.resize(frame, width=800)

    # récupération des dimensions et transformation en collection d'images
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
        0.007843, (300, 300), 127.5)

    # determiner la détection et la prédiction 
    net.setInput(blob)
    detections = net.forward()

    # boucle de détection 
    for i in np.arange(0, detections.shape[2]):
        # calcul de la probabilité de l'objet détecté 
        # en fonction de la prédiction
        confidence = detections[0, 0, i, 2]
        
        # supprimer les détections faibles 
        # inférieures à la probabilité minimale
        if confidence > args["confidence"]:
            # extraire l'index du type d'objet détecté
            # calcul des coordonnées de la fenêtre de détection 
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # creation du contour autour de l'objet détecté
            # insertion de la prédiction de l'objet détecté 
            label = "{}: {:.2f}%".format(CLASSES[idx],
                confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
            # enregistrement de l'image détectée 
            cv2.imwrite("detection.png", frame)
            
            # envoi d'un email avec l'image en pièce jointe
            email = 'votre_e_mail@gmail.com'
            password = 'votre_mot_de_passe'
            send_to_email = 'votre_e_mail@gmail.com'
            subject = 'detection'
            message = 'detection'
            file_location = 'le_répertoire_de_votre_choix/detection.png'
            msg = MIMEMultipart()
            msg['From'] = email
            msg['To'] = send_to_email
            msg['Subject'] = subject
            body = message
            msg.attach(MIMEText(body, 'plain'))
            filename = ntpath.basename(file_location)
            attachment = open(file_location, "rb")
            part = MIMEBase('application', 'octet-stream')
            part.set_payload((attachment).read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', "attachment; filename=%s" % filename)
            msg.attach(part)
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(email, password)
            text = msg.as_string()
            server.sendmail(email, send_to_email, text)
            server.quit()
            
    # affichage du flux vidéo dans une fenètre 

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # la touche q permet d'interrompre la boucle principale
    if key == ord("q"):
        break

    # mise à jour du FPS 
    fps.update()

# arret du compteur et affichage des informations dans la console
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()

##### Pour information, vous pouvez utiliser une webcam USB à la place de la Picamera. Il suffit de modifier cette ligne de code (juste avant la boucle while) :    vs = VideoStream(usePiCamera=True, resolution=(1600, 1200)).start()  en vs = VideoStream(0).start()

#### 4) Mise en situation

Dans le répertoire où sont stockés vos fichiers, ouvrez une console et entrez la commande ci dessous:

$ python3 reconnaissance_objets.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

Au moment de la détection, le programme se charge d’envoyer un e-mail avec une photo en pièce jointe.
