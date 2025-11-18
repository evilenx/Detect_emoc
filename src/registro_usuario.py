import os
from datetime import datetime

CARPETA = "registros_usuarios"

def archivo_usuario(nombre):
    nombre = nombre.lower().replace(" ", "_")
    return os.path.join(CARPETA, nombre + ".txt")

def crear_usuario(nombre):
    """
    Crea el archivo del usuario con solo su nombre y encabezado del registro.
    """
    if not os.path.exists(CARPETA):
        os.makedirs(CARPETA)

    ruta = archivo_usuario(nombre)

    with open(ruta, "w") as f:
        f.write(f"Nombre: {nombre}\n")
        f.write("\n[Registro]\n")   # sección de eventos
    print(f"Usuario '{nombre}' registrado.")

def actualizar_usuario(nombre, emocion_video, emocion_audio):
    """
    Añade una nueva línea al archivo del usuario con fecha/hora y emociones detectadas.
    """
    ruta = archivo_usuario(nombre)

    if not os.path.exists(ruta):
        crear_usuario(nombre)

    ahora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    linea = f"{ahora} | Video: {emocion_video} | Audio: {emocion_audio}\n"

    with open(ruta, "a") as f:
        f.write(linea)
