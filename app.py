# =========================
# 1. IMPORT LIBRARIES
# =========================
# Streamlit builds the web interface.
# OpenAI is used for speech-to-text transcription.
# Sentence Transformers is used for semantic scripture search.
# Sounddevice records audio directly from your computer microphone.

import os
import re
import time
import asyncio
import base64
import json
import websockets

import pandas as pd
import requests
import sounddevice as sd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from scipy.io.wavfile import write
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# 2. LOAD ENVIRONMENT VARIABLES
# AND INITIALIZE OPENAI CLIENT
# =========================
# The API key is stored inside your .env file as OPENAI_API_KEY.

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REALTIME_URL = "wss://api.openai.com/v1/realtime?model=gpt-realtime"

client = OpenAI(api_key=OPENAI_API_KEY)


# =========================
# 3. STREAMLIT PAGE SETUP
# =========================
# This controls the browser page title and layout.

st.set_page_config(page_title="Bible Verse Assistant", layout="wide")

st.title("Bible Verse Assistant")
st.write("Type, upload, or speak a Bible reference and the app will detect and display the verse.")

presentation_mode = st.toggle("Presentation Mode")

if "scripture_history" not in st.session_state:
    st.session_state.scripture_history = []

if "current_displayed_verse" not in st.session_state:
    st.session_state.current_displayed_verse = None

if "scripture_queue" not in st.session_state:
    st.session_state.scripture_queue = []

if "current_reference" not in st.session_state:
    st.session_state.current_reference = None

if "current_display_text" not in st.session_state:
    st.session_state.current_display_text = None

if "reference_history" not in st.session_state:
    st.session_state.reference_history = []

if presentation_mode:
    st.markdown(
        """
        <style>
            header {visibility: hidden;}
            footer {visibility: hidden;}
            .stDeployButton {display: none;}
            [data-testid="stToolbar"] {display: none;}
            [data-testid="stSidebar"] {display: none;}
            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
                max-width: 100%;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

show_controls = st.toggle(
    "Show controls in Presentation Mode",
    value=True
)

translation = "kjv"
user_text = ""
audio_file = None
mic_audio = None
record_seconds = 10

if not presentation_mode or show_controls:

    # =========================
    # 4. USER INPUT SECTION
    # =========================
    # These widgets let the user choose translation, type text,
    # upload sermon audio, or use the browser microphone recorder.

    translation = st.selectbox(
    "Choose Bible Translation",
    ["kjv", "web", "asv", "bbe", "darby", "dra", "ylt"]
)

    compare_translations = st.toggle(
    "Compare Translations",
    value=False
)

    user_text = st.text_area(
        "What did the pastor say?",
        placeholder="Example: Open John chapter 3 verse 16",
        height=150
    )

    audio_file = st.file_uploader(
        "Or upload a short sermon audio clip",
        type=["mp3", "wav", "m4a"]
    )

    mic_audio = st.audio_input("Or record live from your microphone")
# =========================
# 5. AUDIO TRANSCRIPTION FUNCTION
# =========================
# Converts uploaded audio into text using OpenAI transcription.

def transcribe_audio(uploaded_audio):
    file_extension = uploaded_audio.name.split(".")[-1]
    temp_file_path = f"temp_audio_file.{file_extension}"

    with open(temp_file_path, "wb") as f:
        f.write(uploaded_audio.read())

    with open(temp_file_path, "rb") as audio:
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=audio
        )

    return transcript.text


# =========================
# 6. MICROPHONE RECORDING FUNCTION
# =========================
# Records live voice directly from the computer microphone.

def record_from_microphone(seconds=10):
    fs = 44100

    audio = sd.rec(
        int(seconds * fs),
        samplerate=fs,
        channels=1
    )

    sd.wait()

    audio_file_path = "recorded_audio.wav"
    write(audio_file_path, fs, audio)

    return audio_file_path


# =========================
# 7. NUMBER WORD CONVERTER
# =========================
# Converts spoken numbers like "four" or "thirteen" into digits.

def convert_number_words(text):
    text = text.replace("twenty-three", "twenty three")
    text = text.replace("twenty three", "23")
    text = text.replace("twenty two", "22")
    text = text.replace("twenty one", "21")
    text = text.replace("forty-five", "forty five")
    text = text.replace("seventeen forty five", "17 45")
    text = text.replace("seventeen forty-five", "17 45")

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
        "twenty one": "21",
        "twenty two": "22",
        "twenty three": "23",
        "twenty four": "24",
        "twenty five": "25",
        "twenty six": "26",
        "twenty seven": "27",
        "twenty eight": "28",
        "twenty nine": "29",
        "thirty": "30",
        "forty five": "45",
        "forty": "40",
    }

    for word, number in number_words.items():
        text = text.replace(word, number)

    return text

# COMMAND DETECTION

def contains_command(text):
    commands = [
        "let's continue from where we stopped",
        "lets continue from where we stopped",
        "where we stopped",
        "our scripture reading today is",
        "scripture reading today is",
        "let's read",
        "lets read",
        "read",
        "turn to",
        "turn your bibles to",
        "turn our bibles to",
        "open to",
        "open your bibles to",
        "open your bible to",
        "go to",
        "let's go to",
        "lets go to",
        "let's look at",
        "lets look at",
        "look at",
        "the bible says in",
        "as we read in",
        "as it says in",
        "turn with me to",
        "continue from",
        "let's continue from",
        "lets continue from",
        "we are reading from",
        "we will read from",
        "our scripture is",
        "today's scripture is",
        "let's open our bibles to",
        "let open our bibles to",
        "let's open our bible to",
        "let open our bible to",
        "turn to the book of",
        "turn to the book"
        "todays scripture is"
        "scripture open",
        "scripture read",
        "scripture turn",
        "scripture go to",
        "bible open",
        "bible read",
        "bible turn",
        "bible go to",
        "open",
        "so",
    ]

    text = text.lower()

    return any(command in text for command in commands)

def fix_joined_chapter_verse(text):
    valid_books = [
        "genesis", "exodus", "leviticus", "numbers", "deuteronomy",
        "joshua", "judges", "ruth", "samuel", "kings", "chronicles",
        "ezra", "nehemiah", "esther", "job", "psalm", "psalms",
        "proverbs", "ecclesiastes", "isaiah", "jeremiah", "lamentations",
        "ezekiel", "daniel", "hosea", "joel", "amos", "obadiah",
        "jonah", "micah", "nahum", "habakkuk", "zephaniah", "haggai",
        "zechariah", "malachi", "matthew", "mark", "luke", "john",
        "acts", "romans", "corinthians", "galatians", "ephesians",
        "philippians", "colossians", "thessalonians", "timothy",
        "titus", "philemon", "hebrews", "james", "peter", "jude",
        "revelation"
    ]

    for book in valid_books:
        pattern = rf"\b{book}\s+(\d)(\d{{2}})\b"
        replacement = rf"{book} \1:\2"
        text = re.sub(pattern, replacement, text)

    return text

# =========================
# 8. MULTIPLE REFERENCE DETECTION
# =========================
# Extracts one or more Bible references from typed text or transcription.
# Examples:
# - John 3:16
# - Philippians chapter four verse thirteen
# - John 3:16 and Romans 8:28

def detect_multiple_references(text):
    text = text.lower()
    text = text.replace("’", "'")
    text = convert_number_words(text)
    text = fix_joined_chapter_verse(text)

    text = text.replace(",", "")
    text = text.replace(".", "")
    

    text = text.replace("scripture open", "open")
    text = text.replace("bible open", "open")
    text = text.replace("scripture read", "read")
    text = text.replace("bible read", "read")
    text = text.replace("scripture turn", "turn")
    text = text.replace("bible turn", "turn")

    text = text.replace("scripture", "")
    text = text.replace("bible assistant", "")
    text = text.replace("bible", "")

    text = text.replace(" from verse ", " ")
    text = text.replace(" from verses ", " ")
    text = text.replace(" verse ", " ")
    text = text.replace(" verses ", " ")
    text = text.replace(" to ", "-")
    text = text.replace(" and ", "-")
    text = text.replace(" through to ", "-")

    text = text.replace("openjohn", "open john")
    text = text.replace("readjohn", "read john")
    text = text.replace("turnjohn", "turn john")
    text = text.replace("openromans", "open romans")
    text = text.replace("readromans", "read romans")
    text = text.replace("turnromans", "turn romans")

    command_words = [
        "let's continue from where we stopped",
        "lets continue from where we stopped",
        "where we stopped",
        "our scripture reading today is",
        "scripture reading today is",
        "the bible says in",
        "the bible says",
        "now let's read",
        "now let read",
        "let's look at",
        "lets look at",
        "look at",

        "as we read in",
        "as it says in",

        "turn with me to",

        "we are reading from",
        "we're reading from",

        "we are reading",
        "we're reading",

        "reading from",

        "we will read from",

        "our scripture is",

        "today's scripture is",
        "todays scripture is",

        "continue from",
        "let's continue from",
        "lets continue from",
        "let's turn our bibles to",
        "lets turn our bibles to",
        "lets open our bibles to",
        "let's open our bibles to",
        "turn our bibles to",
        "turn your bibles to",
        "turn to the book of",
        "turn to the book",
        "open to the book of",
        "open the book of",
        "book of",
        "open your bibles to",
        "open our bibles to",

        
        "let's read from",
        "lets read from",
        "let us read from",
        "read from",

        "let's read",
        "lets read",
        "lets open",
        "let's open",

        "let's go to",
        "lets go to",
        "let us go to",

        "let's turn to",
        "lets turn to",
        "let us turn to",

        "go to",
        "read from",
        "turn to",
        "open to",

        "chapter",
        "verse",
        "verses",

        "scripture open",
        "scripture read",
        "scripture turn",
        "scripture go to",
        "bible open",
        "bible read",
        "bible turn",
        "bible go to",

        

# SHORT WORDS LAST
        "scripture",
        "bible assistant",
        "bible",
        "read",
        "turn",
        "open",
        "from",
        "to",
        "so"
    ]

    for word in command_words:
        text = text.replace(word, "")

    book_number_words = {
        "first": "1",
        "second": "2",
        "third": "3"
    }

    for word, number in book_number_words.items():
        text = text.replace(word + " ", number + " ")

    text = text.replace("chapter", "")
    text = text.replace("verses", "")
    text = text.replace("verse", "")

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

        bad_prefixes = [
            "To ", "And ", "Bible ", "Bibles ",
            "Our Bible ", "Our Bibles ", "Your Bible ", "Your Bibles "
        ]

        for prefix in bad_prefixes:
            if book.startswith(prefix):
                book = book.replace(prefix, "")

        if book not in valid_books:
            continue

        if verse:
            references.append(f"{book} {chapter}:{verse}")
        else:
            references.append(f"{book} {chapter}")

    return references

# =========================
# 9. BIBLE VERSE FETCH FUNCTION
# =========================
# Retrieves scripture text from bible-api.com.
# The output is formatted verse-by-verse for readability.

def get_verse(reference, translation):
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
                formatted_verses.append(f"**{verse_number}.** {verse_text}")

            return "\n\n".join(formatted_verses)

        return data.get("text", "Verse not found.")

    return f"Verse not found for {reference}. Please check the reference."
    



# =========================
# 10. VERSE DISPLAY FUNCTION
# =========================
# Displays verses normally or in presentation mode.
# Presentation mode is designed for large church/projector screens.

def display_verse(verse_text, presentation_mode):
    if presentation_mode:
        verse_html = verse_text.replace("\n\n", "<br><br>")
        verse_html = verse_html.replace("**", "")

        st.markdown(
            f"""
            <div style="
                min-height: 75vh;
                background-color: #050505;
                color: white;
                padding: 70px;
                border-radius: 24px;
                font-size: 48px;
                line-height: 1.8;
                text-align: center;
                display: flex;
                align-items: center;
                justify-content: center;
            ">
                <div>
                    {verse_html}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(verse_text)

# HELPER FUNCTION. HELPS YOU TO SAVE SCRIPTUES
        
def add_to_history(reference, verse_text):
    st.session_state.scripture_history.append({
        "reference": reference,
        "text": verse_text
    })


def contains_wake_phrase(text):
    text = text.lower()

    wake_phrases = [
        "scripture",
        "bible assistant",
        "scripture",
        "scriptures",
        "structure",
        "bible assistance",
        "bible"
    ]

    return any(phrase in text for phrase in wake_phrases)

def is_previous_scripture_command(text):
    text = text.lower()

    commands = [
        "previous scripture",
        "go back to previous scripture",
        "return to previous scripture",
        "go back to the previous scripture",
        "back to previous scripture",
        "go back",
        "previous passage",
        "previous reference"
    ]

    return any(command in text for command in commands)

def parse_reference(reference):
    match = re.match(r"(.+?)\s+(\d+):(\d+)$", reference)

    if match:
        book = match.group(1)
        chapter = int(match.group(2))
        verse = int(match.group(3))

        return book, chapter, verse

    return None

def get_parallel_verses(reference):
    kjv_text = get_verse(reference, "kjv")
    web_text = get_verse(reference, "web")

    return kjv_text, web_text

def detect_requested_translation(text, default_translation):
    text = text.lower()

    # WEB
    if "web translation" in text:
        return "web"

    if "web version" in text:
        return "web"

    if "world english bible" in text:
        return "web"

    if "from web" in text:
        return "web"

    if "in web" in text:
        return "web"

    # KJV
    if "king james version" in text:
        return "kjv"

    if "king james" in text:
        return "kjv"

    if "kjv" in text:
        return "kjv"

    if "from kjv" in text:
        return "kjv"

    if "in kjv" in text:
        return "kjv"

    if "asv" in text or "american standard" in text:
        return "asv"

    if "bbe" in text or "basic english" in text:
        return "bbe"

    if "darby" in text:
        return "darby"

    if "douay" in text or "dra" in text:
        return "dra"

    if "young" in text or "ylt" in text:
        return "ylt"
    return default_translation

def save_current_reference(reference):

    if "reference_history" not in st.session_state:
        st.session_state.reference_history = []

    current = st.session_state.get("current_reference")

    if current and current != reference:
        st.session_state.reference_history.append(current)

    st.session_state.current_reference = reference


def is_next_verse_command(text):
    text = text.lower()

    next_commands = [
    "next verse",
    "next verses",
    "next scripture",
    "next one",
    "read on",
    "go on",
    "move on",
    "continue",
    "continue reading",
    "continue please",
    "let's continue",
    "lets continue",
    "continue to the next verse",
    "go to the next verse",
    "show the next verse",
    "give me the next verse",
    "verse after that",
    "the next verse",
    "next first",
    "next versus",
    "next vase",
    "next fast",
    "next vest",
    "next vex",
    "next vez",
    "next ves",
    "next vers",
    "next this",
    "go next",
    "nextverse",
    "nes verse",
    "nesverse",
    "nextves",
    "nextvez",
    "next",
    "continue next"
]

    return any(command in text for command in next_commands)


def get_next_reference(reference):
    parts = reference.split()

    book = " ".join(parts[:-1])
    chapter_verse = parts[-1]

    chapter, verse = chapter_verse.split(":")

    next_verse = int(verse) + 1

    return f"{book} {chapter}:{next_verse}"

def add_references_to_queue(references):
    st.session_state.scripture_queue = references[1:]


def has_next_scripture_command(text):
    text = text.lower()

    commands = [
        "next scripture",
        "next reference",
        "next passage",
        "go to next scripture",
        "show next scripture"
    ]

    return any(command in text for command in commands)

def is_previous_verse_command(text):
    text = text.lower()

    commands = [
        "previous verse",
        "go to previous verse",
        "read the previous verse",
        "one verse back",
        "back one verse"
    ]

    return any(command in text for command in commands)


def get_next_scripture_from_queue():
    if st.session_state.scripture_queue:
        return st.session_state.scripture_queue.pop(0)

    return None


async def realtime_scripture_listener():

    st.info("Realtime listener started. Connecting...")

    async with websockets.connect(
        REALTIME_URL,
        additional_headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        },
    ) as websocket:

        st.success("Connected to OpenAI Realtime API")

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
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 300,
                    "create_response": False,
                    "interrupt_response": False
                }
            }
        }
    }
}

        await websocket.send(json.dumps(session_update))

        st.success("Realtime transcription ready")

        transcript_placeholder = st.empty()

        async def receive_events():
            while True:
                message = await websocket.recv()
                event = json.loads(message)

                event_type = event.get("type", "")

                if event_type == "conversation.item.input_audio_transcription.delta":
                    pass
                    #delta = event.get("delta", "")
                    #transcript_placeholder.write(delta)

                elif event_type == "conversation.item.input_audio_transcription.completed":
                    transcript = event.get("transcript", "")

                    st.write("COMPLETED:", transcript)

                    if is_previous_scripture_command(transcript):
                        history = st.session_state.get("reference_history", [])

                        if history:
                            previous_reference = history.pop()

                            verse_text = get_verse(previous_reference, translation)

                            st.session_state.current_reference = previous_reference
                            st.session_state.current_display_reference = previous_reference
                            st.session_state.current_display_text = verse_text

                            st.subheader(previous_reference)
                            display_verse(verse_text, presentation_mode)

                        else:
                            st.warning("No previous scripture found.")

                        return
                    
                    if is_previous_verse_command(transcript):

                        current_reference = st.session_state.get("current_reference")

                        if current_reference:
                            parsed = parse_reference(current_reference)

                        if parsed:
                            book, chapter, verse = parsed

                        if verse > 1:
                            new_reference = f"{book} {chapter}:{verse - 1}"

                            verse_text = get_verse(new_reference, translation)

                            st.session_state.current_reference = new_reference
                            st.session_state.current_display_reference = new_reference
                            st.session_state.current_display_text = verse_text

                            st.subheader(new_reference)
                            display_verse(verse_text, presentation_mode)

                        return

                    references = detect_multiple_references(transcript)

                    if references:
                        reference = references[0]
                        save_current_reference(reference)
                        verse_text = get_verse(reference, translation)

                        st.session_state.current_display_reference = reference
                        st.session_state.current_display_text = verse_text

                        st.subheader(reference)
                        display_verse(verse_text, presentation_mode)

                    

                elif event_type == "error":
                    st.error(event)

        def audio_callback(indata, frames, time, status):
            if status:
                st.warning(status)

            audio_bytes = indata.tobytes()
            audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

            asyncio.run_coroutine_threadsafe(
                websocket.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": audio_base64
                })),
                loop
            )

        loop = asyncio.get_running_loop()

        with sd.InputStream(
            samplerate=24000,
            channels=1,
            dtype="int16",
            callback=audio_callback
        ):
            st.success("Microphone streaming started. Speak now.")
            await receive_events()


# =========================
# 11. SEMANTIC SEARCH MODEL SETUP
# =========================
# Loads the embedding model once and caches it for faster app performance.

@st.cache_resource
def load_semantic_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


semantic_model = load_semantic_model()


# =========================
# 12. LOAD BIBLE DATASET
# =========================
# Loads the full KJV CSV file and prepares references + verse text.

@st.cache_data
def load_bible_data():
    verse_df = pd.read_csv(
        "bible_verses.csv",
        skiprows=4
    )

    verse_df["reference"] = (
        verse_df["Book Name"]
        + " "
        + verse_df["Chapter"].astype(str)
        + ":"
        + verse_df["Verse"].astype(str)
    )

    verse_df = verse_df.rename(columns={"Text": "text"})

    verse_data = verse_df[["reference", "text"]].to_dict("records")
    verse_texts = verse_df["text"].tolist()

    return verse_data, verse_texts


# =========================
# 13. CREATE VERSE EMBEDDINGS
# =========================
# Converts Bible verses into embeddings for semantic topic search.

@st.cache_data
def create_embeddings(texts):
    return semantic_model.encode(texts)


verse_data, verse_texts = load_bible_data()
verse_embeddings = create_embeddings(verse_texts)


# =========================
# 14. SEMANTIC SCRIPTURE SEARCH FUNCTION
# =========================
# Searches for scriptures by meaning/topic, not only exact references.

def semantic_scripture_search(query, top_k=20):
    query_embedding = semantic_model.encode([query])

    similarities = cosine_similarity(
        query_embedding,
        verse_embeddings
    )[0]

    top_indices = similarities.argsort()[-top_k:][::-1]

    results = []

    for index in top_indices:
        results.append({
            "reference": verse_data[index]["reference"],
            "text": verse_data[index]["text"],
            "score": round(float(similarities[index]) * 100, 2)
        })

    return results


# =========================
# 15. VOICE SCRIPTURE DETECTION UI
# =========================
# Manual one-click recording: records, transcribes, detects, and displays scripture.

# Show currently displayed scripture
record_seconds = st.slider(
    "Recording length",
    min_value=5,
    max_value=30,
    value=10
)

if st.button("Record and Find Scripture", key="record_voice"):
    with st.spinner("Recording... Speak now."):
        recorded_file = record_from_microphone(record_seconds)

    with st.spinner("Transcribing voice..."):
        with open(recorded_file, "rb") as audio:
            transcript = client.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",
                file=audio
            )

    final_text = transcript.text

    st.subheader("Transcription")
    st.write(final_text)

    if contains_command(final_text):
        references = detect_multiple_references(final_text)
    else:
        references = []

    if references:
        st.subheader("Detected References")

        for reference in references:
            st.write(reference)

            if compare_translations:
                kjv_text, web_text = get_parallel_verses(reference)

                add_to_history(
                    reference,
                    f"KJV:\n{kjv_text}\n\nWEB:\n{web_text}"
                )

                st.subheader(reference)

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### KJV")
                    display_verse(kjv_text, presentation_mode)

                with col2:
                    st.markdown("### WEB")
                    display_verse(web_text, presentation_mode)

            else:
                requested_translation = detect_requested_translation(
                    final_text,
                    translation
                )

                verse_text = get_verse(
                    reference,
                    requested_translation
                )

                st.session_state.current_display_reference = reference
                st.session_state.current_display_text = verse_text

                add_to_history(reference, verse_text)

                st.subheader(reference)
                save_current_reference(reference)
                st.session_state.current_displayed_verse = verse_text
                display_verse(verse_text, presentation_mode)

    else:
        st.warning("No Bible reference detected.")

# =========================
# 16. CONTINUOUS LISTENING MODE
# =========================
# Repeatedly records short audio chunks, transcribes them,
# detects Bible references, and refreshes automatically.

st.subheader("Continuous Listening Mode")

continuous_mode = st.checkbox("Enable continuous listening")

realtime_mode = st.checkbox(
    "Enable Realtime Scripture Detection"
)

wake_phrase_mode = st.toggle(
    "Wake Phrase Mode",
    value=False
)

listening_seconds = st.slider(
    "Listening interval in seconds",
    min_value=2,
    max_value=10,
    value=2,
    key="continuous_seconds"
)

if continuous_mode and realtime_mode:
    st.error(
        "Please enable either Continuous Listening or Realtime Mode, not both."
    )
    st.stop()

if realtime_mode:
    asyncio.run(realtime_scripture_listener())

if continuous_mode:
    st.info("Continuous listening is active. Speak a Bible reference clearly.")

    if st.session_state.current_display_text:
        st.subheader(st.session_state.current_display_reference)
        display_verse(
            st.session_state.current_display_text,
            presentation_mode
        )

    placeholder = st.empty()
    scripture_placeholder = st.empty()

    with st.spinner("Listening..."):
        recorded_file = record_from_microphone(listening_seconds)

        try:
            with open(recorded_file, "rb") as audio:
                transcript = client.audio.transcriptions.create(
                    model="gpt-4o-mini-transcribe",
                    file=audio
        )

            final_text = transcript.text

        except Exception as e:
            st.warning("Could not connect to transcription service. Please try again.")
            time.sleep(1)
            st.rerun()

    final_text = transcript.text
    placeholder.write(f"Transcription: {final_text}")

    if wake_phrase_mode and not contains_wake_phrase(final_text):
        st.info("Wake phrase not detected. Ignoring this audio.")
        time.sleep(0.2)
        st.rerun()

    st.write("Current reference:", st.session_state.current_reference)
    st.write("Next command detected:", is_next_verse_command(final_text))

    # NEXT SCRIPTURE COMMAND
    if has_next_scripture_command(final_text):
        next_reference = get_next_scripture_from_queue()

        if next_reference:
            requested_translation = detect_requested_translation(
                final_text,
                translation
            )

            verse_text = get_verse(
                next_reference,
                requested_translation
            )

            add_to_history(next_reference, verse_text)

            with scripture_placeholder.container():
                st.session_state.current_display_reference = next_reference
                st.session_state.current_display_text = verse_text

                st.subheader(next_reference)
                save_current_reference(next_reference)
                st.session_state.current_displayed_verse = verse_text
                display_verse(verse_text, presentation_mode)

            time.sleep(1)
            st.rerun()

    # NEXT VERSE COMMAND
    if is_next_verse_command(final_text) and st.session_state.current_reference:
        next_reference = get_next_reference(st.session_state.current_reference)

        requested_translation = detect_requested_translation(
            final_text,
            translation
        )

        verse_text = get_verse(
            next_reference,
            requested_translation
        )

        add_to_history(next_reference, verse_text)

        with scripture_placeholder.container():
            st.session_state.current_display_reference = next_reference
            st.session_state.current_display_text = verse_text

            st.subheader(next_reference)
            save_current_reference(next_reference)
            st.session_state.current_displayed_verse = verse_text
            display_verse(verse_text, presentation_mode)

        time.sleep(1)
        st.rerun()

    # NORMAL SCRIPTURE DETECTION
    if contains_command(final_text):
        references = detect_multiple_references(final_text)
    else:
        references = []

    if references:
        add_references_to_queue(references)

        reference = references[0]

        st.subheader("Current Scripture")
        st.write(reference)

        if compare_translations:
            kjv_text, web_text = get_parallel_verses(reference)

            add_to_history(
                reference,
                f"KJV:\n{kjv_text}\n\nWEB:\n{web_text}"
            )

            with scripture_placeholder.container():
                st.session_state.current_display_reference = reference
                st.session_state.current_display_text = f"KJV:\n{kjv_text}\n\nWEB:\n{web_text}"

                st.subheader(reference)

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### KJV")
                    display_verse(kjv_text, presentation_mode)

                with col2:
                    st.markdown("### WEB")
                    display_verse(web_text, presentation_mode)

                save_current_reference(reference)

        else:
            requested_translation = detect_requested_translation(
                final_text,
                translation
            )

            verse_text = get_verse(
                reference,
                requested_translation
            )

            add_to_history(reference, verse_text)

            with scripture_placeholder.container():
                st.session_state.current_display_reference = reference
                st.session_state.current_display_text = verse_text

                st.subheader(reference)
                save_current_reference(reference)
                st.session_state.current_displayed_verse = verse_text
                display_verse(verse_text, presentation_mode)

        if st.session_state.scripture_queue:
            st.subheader("Scripture Queue")
            for queued_reference in st.session_state.scripture_queue:
                st.write(queued_reference)

    else:
        st.warning("No Bible reference detected.")

    time.sleep(1)
    st.rerun()

# =========================
# 17. TEXT AND AUDIO FILE VERSE SEARCH
# =========================
# Handles typed Bible references and uploaded sermon audio files.

if st.button("Find Verse"):
    if audio_file is not None:
        with st.spinner("Transcribing uploaded audio..."):
            final_text = transcribe_audio(audio_file)

        st.subheader("Audio File Transcription")
        st.write(final_text)
    else:
        final_text = user_text

    references = detect_multiple_references(final_text)

    if references:
        st.subheader("Detected References")

        for reference in references:
            st.write(reference)

            if compare_translations:
                kjv_text, web_text = get_parallel_verses(reference)

                add_to_history(reference, f"KJV:\n{kjv_text}\n\nWEB:\n{web_text}")

                st.subheader(reference)

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### KJV")
                    display_verse(kjv_text, presentation_mode)

                with col2:
                    st.markdown("### WEB")
                    display_verse(web_text, presentation_mode)

            else:
                requested_translation = detect_requested_translation(
                final_text,
                translation
                )

                verse_text = get_verse(
                reference,
                requested_translation
                )


                add_to_history(reference, verse_text)

                st.subheader(reference)
                save_current_reference(reference)
                st.session_state.current_displayed_verse = verse_text
                display_verse(verse_text, presentation_mode)

    else:
        st.warning("No Bible reference detected.")


# =========================
# 18. SEMANTIC SCRIPTURE SEARCH UI
# =========================
# Allows the user to search the full Bible by topic or meaning.

st.subheader("Semantic Scripture Search")

topic_query = st.text_input(
    "Search scriptures by topic",
    placeholder="Example: scriptures about faith, healing, peace, fear, trust"
)

if st.button("Search Scriptures"):
    if topic_query.strip() == "":
        st.warning("Please enter a topic to search.")
    else:
        results = semantic_scripture_search(topic_query)

        for result in results:
            st.subheader(result["reference"])
            st.write(f"Similarity: {result['score']}%")

            st.session_state.current_displayed_verse = result["text"]

            display_verse(result["text"], presentation_mode)

# =========================
# 19. RECENT SCRIPTURE HISTORY
# =========================
# Stores and displays recently detected scriptures.

st.subheader("Recent Scriptures")

if st.session_state.scripture_history:
    for item in reversed(st.session_state.scripture_history[-10:]):
        with st.expander(item["reference"]):
            st.markdown(item["text"])
else:
    st.write("No scriptures detected yet.")
