import asyncio
import base64
import json
import os

import requests
import sounddevice as sd
import websockets
from dotenv import load_dotenv


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

REALTIME_URL = "wss://api.openai.com/v1/realtime?model=gpt-realtime"


def convert_number_words(text):

    text = text.replace("twenty-three", "twenty three")
    text = text.replace("twenty three", "23")
    text = text.replace("thirty nine", "39")
    text = text.replace("thirty-nine", "39")
    text = text.replace("five to six", "5-6")
    text = text.replace("five and six", "5-6")
    number_words = {
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
        "eleven": "11",
        "twelve": "12",
        "thirteen": "13",
        "fourteen": "14",
        "fifteen": "15",
        "sixteen": "16",
        "seventeen": "17",
        "eighteen": "18",
        "nineteen": "19",
        "twenty": "20",
    }

    for word, number in number_words.items():
        text = text.replace(word, number)

    return text


def detect_multiple_references(text):
    import re

    text = text.lower()
    text = text.replace("’", "'")
    text = convert_number_words(text)

    text = text.replace(",", "")
    text = text.replace(".", "")

    text = text.replace(" from verse ", " ")
    text = text.replace(" from verses ", " ")
    text = text.replace(" verse ", " ")
    text = text.replace(" verses ", " ")
    text = text.replace(" to ", "-")
    text = text.replace(" and ", "-")

    command_words = [
        "let's read",
        "lets read",
        "read",
        "open",
        "turn to",
        "go to",
        "chapter",
        "verse",
        "verses",
        "from",
        "to",
        "let's look at",
        "lets look at",
        "look at"
    ]

    for word in command_words:
        text = text.replace(word, "")

    text = " ".join(text.split())

    valid_books = [
        "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy",
        "Joshua", "Judges", "Ruth", "1 Samuel", "2 Samuel",
        "1 Kings", "2 Kings", "1 Chronicles", "2 Chronicles",
        "Ezra", "Nehemiah", "Esther", "Job", "Psalm", "Psalms",
        "Proverbs", "Ecclesiastes", "Song Of Solomon", "Isaiah",
        "Jeremiah", "Lamentations", "Ezekiel", "Daniel", "Hosea",
        "Joel", "Amos", "Obadiah", "Jonah", "Micah", "Nahum",
        "Habakkuk", "Zephaniah", "Haggai", "Zechariah", "Malachi",
        "Matthew", "Mark", "Luke", "John", "Acts", "Romans",
        "1 Corinthians", "2 Corinthians", "Galatians", "Ephesians",
        "Philippians", "Colossians", "1 Thessalonians", "2 Thessalonians",
        "1 Timothy", "2 Timothy", "Titus", "Philemon", "Hebrews",
        "James", "1 Peter", "2 Peter", "1 John", "2 John", "3 John",
        "Jude", "Revelation"
    ]

    pattern = r"([1-3]?\s?[a-z]+(?:\s+[a-z]+)?)\s+(\d+)(?:[:\s]+(\d+(?:-\d+)?))?"

    
    matches = re.findall(pattern, text)

    references = []

    for match in matches:
        book = match[0].title().strip()
        chapter = match[1]
        verse = match[2]

        if book not in valid_books:
            continue

        if verse:
            references.append(f"{book} {chapter}:{verse}")
        else:
            references.append(f"{book} {chapter}")

    return references

def get_verse(reference, translation="kjv"):
    url = f"https://bible-api.com/{reference}?translation={translation}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        verses = data.get("verses", [])

        if verses:
            formatted_verses = []

            for verse in verses:
                verse_number = verse.get("verse")
                verse_text = verse.get("text", "").strip()
                formatted_verses.append(f"{verse_number}. {verse_text}")

            return "\n".join(formatted_verses)

        return data.get("text", "Verse not found.")

    return "Verse not found."

async def realtime_listener():
    async with websockets.connect(
        REALTIME_URL,
        additional_headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            
        },
    ) as websocket:

        print("Connected to OpenAI Realtime API")

        session_update = {
    "type": "session.update",
    "session": {
        "type": "realtime",
        "audio": {
            "input": {
                "format": {
                    "type": "audio/pcm",
                    "rate": 24000
                },
                "transcription": {
                    "model": "gpt-4o-mini-transcribe",
                    "language": "en"
                }
            }
        }
    }
}

        await websocket.send(json.dumps(session_update))
        print("Session update sent")
        print("Speak into your microphone...")

        async def receive_events():
            while True:
                message = await websocket.recv()
                event = json.loads(message)

                event_type = event.get("type", "")

                if event_type == "conversation.item.input_audio_transcription.delta":
                     print(event.get("delta", ""), end="", flush=True)

                elif event_type == "conversation.item.input_audio_transcription.completed":
                    transcript = event.get("transcript", "")
                    print("\nCOMPLETED:", transcript)

                    references = detect_multiple_references(transcript)

                    if references:
                        print("SCRIPTURE FOUND:", references)
                        first_reference = references[0]
                        verse_text = get_verse(first_reference, "kjv")

                        print("\nREFERENCE:", first_reference)
                        print("VERSE:")
                        print(verse_text)

                elif event_type == "error":
                     
                    print("ERROR:", event)

        def audio_callback(indata, frames, time, status):
            if status:
                print(status)

            audio_bytes = indata.tobytes()
            audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

            asyncio.run_coroutine_threadsafe(
                websocket.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": audio_base64
                })),
                loop
            )

        with sd.InputStream(
            samplerate=24000,
            channels=1,
            dtype="int16",
            callback=audio_callback
        ):
            await receive_events()


loop = asyncio.get_event_loop()
loop.run_until_complete(realtime_listener())