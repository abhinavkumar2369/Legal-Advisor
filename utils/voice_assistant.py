import sounddevice as sd
import numpy as np
import speech_recognition as sr
import scipy.io.wavfile as wav
import tempfile
import pyttsx3
import os

def record_audio(duration=5, fs=44100):
    """Records audio from the microphone."""
    print("🎤 Recording...")
    try:
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()
        return audio, fs
    except Exception as e:
        print(f"Recording failed: {e}")
        return None, None

def save_audio_to_wav(audio, fs):
    """Saves numpy array audio to a WAV file and returns file path."""
    try:
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        wav.write(tmp_file.name, fs, audio)
        return tmp_file.name
    except Exception as e:
        print(f"Saving audio failed: {e}")
        return None

def transcribe_audio(file_path):
    """Transcribes audio using Google Web Speech API."""
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(file_path) as source:
            audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        return "❌ Could not understand audio."
    except sr.RequestError:
        return "❌ Google Speech API unavailable."
    except Exception as e:
        return f"❌ Error: {e}"

def speak_text(text):
    """Speaks text using pyttsx3 (offline TTS)."""
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 170)
        engine.setProperty('volume', 1.0)
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"Text-to-speech failed: {e}")
