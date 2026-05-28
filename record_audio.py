import sounddevice as sd
from scipy.io.wavfile import write
from dotenv import load_dotenv
from openai import OpenAI
import os
import requests
import re

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# -----------------------------
# RECORD AUDIO
# -----------------------------

fs = 44100
seconds = 15

print("Recording for 15 seconds. Start speaking now.")

audio = sd.rec(
    int(seconds * fs),
    samplerate=fs,
    channels=1
)

sd.wait()

audio_file = "recorded_audio.wav"

write(audio_file, fs, audio)
print("Audio file size:", os.path.getsize(audio_file), "bytes")

print("Recording saved.")

# -----------------------------
# TRANSCRIBE AUDIO
# -----------------------------

with open(audio_file, "rb") as file:

    transcript = client.audio.transcriptions.create(
        model="gpt-4o-mini-transcribe",
        file=file
    )

transcribed_text = transcript.text

print("\nTranscription:")
print(transcribed_text)

# -----------------------------
# DETECT BIBLE REFERENCE
# -----------------------------

def detect_reference(text):

    text = text.lower()

    text = text.replace(",", "")
    text = text.replace(".", "")

    command_words = [
    "let's open our bibles to",
    "lets open our bibles to",
    "let's turn our bibles to",
    "lets turn our bibles to",
    "turn our bibles to",
    "turn your bibles to",
    "open your bibles to",
    "open our bibles to",
    "open to",
    "open",
    "turn to",
    "turn",
    "read from",
    "read",
    "let's read",
    "lets read",
    "go to",
    "from",
    "to"
]

    for word in command_words:
        text = text.replace(word, "")

    text = text.replace("first ", "1 ")
    text = text.replace("second ", "2 ")
    text = text.replace("third ", "3 ")

    text = text.replace("chapter", "")
    text = text.replace("verses", "")
    text = text.replace("verse", "")

    text = " ".join(text.split())

    pattern = r"([1-3]?\s?[a-z]+(?:\s+[a-z]+)?)\s+(\d+)(?:\s+(\d+(?:-\d+)?))?"

    match = re.search(pattern, text)

    if match:

        book = match.group(1).title().strip()
        chapter = match.group(2)
        verse = match.group(3)

        if verse:
            return f"{book} {chapter}:{verse}"
        else:
            return f"{book} {chapter}"

    return None

reference = detect_reference(transcribed_text)

print("\nDetected Reference:")
print(reference)

# -----------------------------
# FETCH SCRIPTURE
# -----------------------------

if reference:

    url = f"https://bible-api.com/{reference}?translation=kjv"

    response = requests.get(url)

    if response.status_code == 200:

        data = response.json()

        print("\nScripture:\n")

        for verse in data["verses"]:

            print(
                f"{verse['verse']}. {verse['text']}"
            )

    else:
        print("Verse not found.")

else:
    print("No Bible reference detected.")