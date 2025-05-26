# voice_query.py
import sounddevice as sd
import scipy.io.wavfile as wav
import os
from faster_whisper import WhisperModel
from query_data import main as run_query
import sys

# Record audio
def record_audio(filename="question.wav", duration=5):
    print("🎤 Speak your question...")
    fs = 16000
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    wav.write(filename, fs, audio)
    print("✅ Recorded.")

# Transcribe with Whisper
def transcribe_audio(filename="question.wav"):
    model = WhisperModel("base.en", compute_type="int8")
    segments, _ = model.transcribe(filename)
    text = "".join([seg.text for seg in segments])
    print(f"📝 You asked: {text.strip()}")
    return text.strip()

# Run your existing query script
def ask_question_from_audio():
    record_audio()
    question = transcribe_audio()
    sys.argv = ["query_data.py", question]  # simulate CLI call
    run_query()

if __name__ == "__main__":
    ask_question_from_audio()