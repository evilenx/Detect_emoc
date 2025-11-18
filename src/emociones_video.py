from fer import FER
import cv2

detector = FER(mtcnn=False)

def emocion_por_video(frame):
    resultados = detector.detect_emotions(frame)
    if not resultados:
        return None, 0.0
   
    datos = resultados[0]
    emociones = datos["emotions"]
    emocion = max(emociones, key=emociones.get)
    confianza = emociones[emocion]

    # extra: dibuja recuadro y texto
    (x, y, w, h) = datos["box"]
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255,255,255), 2)
    cv2.putText(frame, emocion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    return emocion, confianza

