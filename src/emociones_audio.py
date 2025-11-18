import numpy as np
import librosa

def emocion_por_audio(audio, sr=16000):
    energia = np.mean(audio**2)
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio)[0])
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=5)
   
    # Algoritmo simple basado en análisis de características
    if energia < 0.01:
        emocion = "neutral"
        conf = 0.60
    elif energia > 0.05 and zcr > 0.08:
        emocion = "enojado"
        conf = 0.75
    elif zcr < 0.04:
        emocion = "triste"
        conf = 0.70
    else:
        emocion = "feliz"
        conf = 0.65

    return emocion, conf
