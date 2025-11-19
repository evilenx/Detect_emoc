import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import time
import csv
from datetime import datetime

from emociones_video import emocion_por_video
from emociones_audio import emocion_por_audio
from captura_audio import grabar_audio
from reconocimiento_persona import (
    recolectar_rostro,
    entrenar_lbph,
    cargar_modelo,
    reconocer
)
from registro_usuario import crear_usuario, actualizar_usuario
# Si ya tienes registro_estadisticas.py:
# from registro_estadisticas import actualizar_estadisticas

CARPETA_FACES = "data_faces"
CARPETA_REGISTROS = "registros_usuarios"

# ---------------------------------------------------
# Utilidades
# ---------------------------------------------------

def guardar_csv(nombre, emo_v, conf_v, emo_a, conf_a, fps=0):
    with open("registro_emociones.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            nombre,
            emo_v,
            conf_v,
            emo_a,
            conf_a,
            fps
        ])

def asegurar_carpetas():
    if not os.path.exists(CARPETA_FACES):
        os.makedirs(CARPETA_FACES)
    if not os.path.exists(CARPETA_REGISTROS):
        os.makedirs(CARPETA_REGISTROS)

asegurar_carpetas()

st.set_page_config(page_title="Detector Multimodal", layout="wide")
st.title("HumanFeel AI")


# Sidebar: modo de trabajo
modo = st.sidebar.radio(
    "Selecciona modo:",
    ["Registrar usuario", "Reconocer & analizar emociones"]
)

# ---------------------------------------------------
# MODO 1: REGISTRAR USUARIO
# ---------------------------------------------------
if modo == "Registrar usuario":
    st.subheader("🧾 Registro de nuevo usuario")

    nombre = st.text_input("Nombre de la persona (ej: Gabriella Herrera):")

    st.write("Toma una foto clara del rostro (mirando a la cámara):")
    foto = st.camera_input("Captura una foto")

    if st.button("📸 Guardar rostro y entrenar modelo"):
        if not nombre:
            st.error("Por favor, ingresa un nombre antes de registrar.")
        elif not foto:
            st.error("Por favor, toma una foto del rostro.")
        else:
            # Convertir imagen de Streamlit a frame (BGR para OpenCV)
            img = Image.open(foto)
            img_np = np.array(img)  # RGB
            frame = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            # Recolectar rostro (guarda en data_faces/nombre/)
            ok = recolectar_rostro(frame, nombre)
            if not ok:
                st.error("No se detectó ningún rostro en la imagen. Intenta nuevamente.")
            else:
                # Entrenar LBPH con todos los usuarios
                model, label_map = entrenar_lbph()
                if model is None:
                    st.error("No se pudo entrenar el modelo. Verifica que existan imágenes.")
                else:
                    # Crear archivo de usuario
                    crear_usuario(nombre)
                    st.success(f"Usuario '{nombre}' registrado y modelo LBPH entrenado.")

# ---------------------------------------------------
# MODO 2: RECONOCER & ANALIZAR EMOCIONES
# ---------------------------------------------------
elif modo == "Reconocer & analizar emociones":
    st.subheader("Reconocimiento facial + emociones en tiempo real")

    st.write("Al iniciar, se intentará reconocer el rostro con el modelo LBPH entrenado.")
    iniciar = st.button("▶ Iniciar cámara y análisis")

    if iniciar:
        model, label_map = cargar_modelo()
        if model is None:
            st.error("No hay modelo entrenado. Primero registra al menos un usuario en la pestaña 'Registrar usuario'.")
        else:
            video_placeholder = st.empty()
            info_placeholder = st.empty()

            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("No se pudo abrir la cámara.")
            else:
                st.warning("Para detener, cierra la pestaña o detén Streamlit en la terminal (Ctrl+C).")

            ultimo_audio = time.time()
            emo_audio_actual = "Analizando..."
            conf_audio_actual = 0.0

            while True:
                ret, frame = cap.read()
                if not ret:
                    st.error("No se pudo leer frame de la cámara.")
                    break

                # Reconocimiento facial
                nombre, conf_id = reconocer(frame, model, label_map)
                if nombre is None:
                    nombre = "desconocido"

                # Emociones por video
                emo_v, conf_v = emocion_por_video(frame)

                # Emociones por audio cada 2s
                if time.time() - ultimo_audio > 2:
                    audio = grabar_audio()
                    emo_audio_actual, conf_audio_actual = emocion_por_audio(audio)
                    ultimo_audio = time.time()

                # Actualizar registros solo si NO es desconocido
                if nombre != "desconocido":
                    actualizar_usuario(nombre, emo_v, emo_audio_actual)
                    # Si usas estadísticas:
                    # actualizar_estadisticas(nombre, emo_v, emo_audio_actual)
                    guardar_csv(nombre, emo_v, conf_v, emo_audio_actual, conf_audio_actual, fps=0)

                # Mostrar info en pantalla
                info_placeholder.markdown(
                    f"""
                    **Usuario:** `{nombre}` |  
                    **Video:** {emo_v} ({conf_v:.2f})  
                    **Audio:** {emo_audio_actual} ({conf_audio_actual:.2f})  
                    """
                )

                # Mostrar frame en Streamlit
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                video_placeholder.image(img, width='stretch')
                #video_placeholder.image(img, use_container_width=True)

            cap.release()

