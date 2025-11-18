import sounddevice as sd
import numpy as np

def grabar_audio(duracion=1.0, samplerate=16000):
  audio = sd.rec(int(duracion * samplerate), samplerate=samplerate, channels=1, dtype='float32')
  sd.wait()
  return audio.flatten()
