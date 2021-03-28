------------------------------------------------------------------------------------------------------------------------------------------------
## TensorFlow Lite
Solution 1: Choix de l'open source: **TensorFlow** 

- https://www.tensorflow.org/lite

### Doc technique

#### Détection d'objets
- https://www.tensorflow.org/lite/models/object_detection/overview?hl=fr

#### Doc d'installation
- https://towardsdatascience.com/real-time-object-tracking-with-tensorflow-raspberry-pi-and-pan-tilt-hat-2aeaef47e134

### Software Installation
#### 1) Install system dependencies

$ sudo apt-get update && sudo apt-get install -y python3-dev libjpeg-dev libatlas-base-dev raspi-gpio libhdf5-dev python3-smbus

#### 2) Create a new project directory
$ mkdir rpi-deep-pantilt && cd rpi-deep-pantilt

#### 3) Create a new virtual environment
$ python3 -m venv .venv

#### 4) Activate the virtual environment
$ source .venv/bin/activate && python3 -m pip install --upgrade pip
#### 5) Install TensorFlow 2.0 from a community-built wheel.
$ pip install https://github.com/leigh-johnson/Tensorflow-bin/blob/master/tensorflow-2.0.0-cp37-cp37m-linux_armv7l.whl?raw=true

#### 6) Install the rpi-deep-pantilt Python package
$ python3 -m pip install rpi-deep-pantilt


## TensorFlow Lite
Solution 2: Choix de l'open source: **TensorFlow Lite**

#### Doc d'installation
- https://www.tensorflow.org/lite/guide/build_rpi

### Étape 1. Clonez la chaîne d'outils de compilation croisée officielle de Raspberry Pi
git clone https://github.com/raspberrypi/tools.git rpi_tools

### Étape 2. Cloner le référentiel TensorFlow
git clone https://github.com/tensorflow/tensorflow.git tensorflow_src


### Étape 3. Exécutez le script suivant à la racine du référentiel TensorFlow pour télécharger toutes les dépendances de construction:
cd tensorflow_src && ./tensorflow/lite/tools/make/download_dependencies.sh

### Étape 4a. Pour construire le binaire ARMv7 pour Raspberry Pi 2, 3 et 4
PATH=../rpi_tools/arm-bcm2708/arm-rpi-4.9.3-linux-gnueabihf/bin:$PATH \
  ./tensorflow/lite/tools/make/build_rpi_lib.sh
  
Vous pouvez ajouter des options Make supplémentaires ou des noms de cible au script build_rpi_lib.sh car il s'agit d'un wrapper de Make avec TFLite Makefile . Voici quelques options possibles:


./tensorflow/lite/tools/make/build_rpi_lib.sh clean # clean object files
./tensorflow/lite/tools/make/build_rpi_lib.sh -j 16 # run with 16 jobs to leverage more CPU cores
./tensorflow/lite/tools/make/build_rpi_lib.sh label_image # # build label_image binary

### Étape 4b. Pour construire le binaire ARMv6 pour Raspberry Pi Zero

PATH=../rpi_tools/arm-bcm2708/arm-rpi-4.9.3-linux-gnueabihf/bin:$PATH \
  ./tensorflow/lite/tools/make/build_rpi_lib.sh TARGET_ARCH=armv6
Remarque: Cela devrait compiler une bibliothèque statique dans: tensorflow/lite/tools/make/gen/rpi_armv6/lib/libtensorflow-lite.a .

Compilez nativement sur Raspberry Pi
Les instructions suivantes ont été testées sur Raspberry Pi Zero, Raspberry Pi OS GNU / Linux 10 (Buster), gcc version 8.3.0 (Raspbian 8.3.0-6 + rpi1):

**Pour compiler nativement TensorFlow Lite, procédez comme suit:**

### Étape 1. Connectez-vous à votre Raspberry Pi et installez la chaîne d'outils
sudo apt-get install build-essential

### Étape 2. Cloner le référentiel TensorFlow
git clone https://github.com/tensorflow/tensorflow.git tensorflow_src

### Étape 3. Exécutez le script suivant à la racine du référentiel TensorFlow pour télécharger toutes les dépendances de construction
cd tensorflow_src && ./tensorflow/lite/tools/make/download_dependencies.sh
Remarque: vous ne devez le faire qu'une seule fois.

### Étape 4. Vous devriez alors être en mesure de compiler TensorFlow Lite avec:
./tensorflow/lite/tools/make/build_rpi_lib.sh
Remarque: Cela devrait compiler une bibliothèque statique dans: tensorflow/lite/tools/make/gen/lib/rpi_armv6/libtensorflow-lite.a .
Compilation croisée pour armhf avec Bazel
Vous pouvez utiliser les chaînes d' outils ARM GCC avec Bazel pour créer une bibliothèque partagée armhf compatible avec Raspberry Pi 2, 3 et 4.

Remarque: La bibliothèque partagée générée nécessite la glibc 2.28 ou une version supérieure pour s'exécuter.
Les instructions suivantes ont été testées sur Ubuntu 16.04.3 PC 64 bits (AMD64) et TensorFlow devel image docker tensorflow / tensorflow: devel .

**Pour effectuer une compilation croisée de TensorFlow Lite avec Bazel, procédez comme suit:**

### Étape 1. Installez Bazel
Bazel est le système de construction principal pour TensorFlow. Installez la dernière version du système de build Bazel .
Remarque: si vous utilisez l'image Docker TensorFlow, Bazel est déjà disponible.

### Étape 2. Cloner le référentiel TensorFlow
git clone https://github.com/tensorflow/tensorflow.git tensorflow_src
Remarque: Si vous utilisez l'image Docker TensorFlow, le dépôt est déjà fourni dans /tensorflow_src/ .

### Étape 3. Construisez le binaire ARMv7 pour Raspberry Pi 2, 3 et 4
Bibliothèque C

bazel build --config=elinux_armhf -c opt //tensorflow/lite/c:libtensorflowlite_c.so
Consultez la page API C de TensorFlow Lite pour plus de détails.

Bibliothèque C ++

bazel build --config=elinux_armhf -c opt //tensorflow/lite:libtensorflowlite.so
