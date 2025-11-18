import os
import cv2
import numpy as np

CARPETA_FACES = "data_faces"
CARPETA_MODELOS = "models"
MODELO_RUTA = os.path.join(CARPETA_MODELOS, "lbph_model.xml")

# Detecttor de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def asegurar_carpetas():
    if not os.path.exists(CARPETA_FACES):
        os.makedirs(CARPETA_FACES)
    if not os.path.exists(CARPETA_MODELOS):
        os.makedirs(CARPETA_MODELOS)


def recolectar_rostro(frame, nombre):
    asegurar_carpetas()
    nombre = nombre.lower().replace(" ", "_")
    carpeta_usuario = os.path.join(CARPETA_FACES, nombre)

    if not os.path.exists(carpeta_usuario):
        os.makedirs(carpeta_usuario)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rostros = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in rostros:
        rostro = gray[y:y+h, x:x+w]
        idx = len(os.listdir(carpeta_usuario)) + 1
        ruta_foto = os.path.join(carpeta_usuario, f"{idx}.jpg")
        cv2.imwrite(ruta_foto, rostro)
        return True

    return False


def entrenar_lbph():
    asegurar_carpetas()

    labels = []
    faces = []
    label_id = 0
    label_map = {}

    for carpeta in os.listdir(CARPETA_FACES):
        ruta = os.path.join(CARPETA_FACES, carpeta)
        if os.path.isdir(ruta):
            label_map[label_id] = carpeta
            for imagen in os.listdir(ruta):
                img_ruta = os.path.join(ruta, imagen)
                img = cv2.imread(img_ruta, cv2.IMREAD_GRAYSCALE)
                faces.append(img)
                labels.append(label_id)
            label_id += 1

    if len(faces) == 0:
        return None, None

    faces = np.array(faces)
    labels = np.array(labels)

    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(faces, labels)
    model.save(MODELO_RUTA)

    return model, label_map


def cargar_modelo():
    if os.path.exists(MODELO_RUTA):
        model = cv2.face.LBPHFaceRecognizer_create()
        model.read(MODELO_RUTA)

        # reconstruir label_map
        label_map = {}
        for idx, carpeta in enumerate(os.listdir(CARPETA_FACES)):
            label_map[idx] = carpeta

        return model, label_map

    return None, None


def reconocer(frame, model, label_map):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rostros = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in rostros:
        rostro = gray[y:y+h, x:x+w]
        label, confianza = model.predict(rostro)

        if confianza < 80:   # valor recomendado
            nombre = label_map.get(label, "desconocido")
            return nombre, 1.0 - confianza / 100.0

    return None, 0.0


