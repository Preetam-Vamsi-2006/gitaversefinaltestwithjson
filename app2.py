import streamlit as st
import json
import os
import re
import unicodedata
from gtts import gTTS
from mtranslate import translate
import tempfile

# ======================================================
# CONFIG
# ======================================================
DATA_DIR = "BhagavatGitaJsonFiles"

LANGUAGE_CODES = {
    "English": "en",
    "Hindi": "hi",
    "Telugu": "te",
    "Tamil": "ta",
    "Malayalam": "ml",
    "Kannada": "kn"
}

# ======================================================
# NORMALIZE SANSKRIT (DO NOT CHANGE)
# ======================================================
def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.replace("\n", " ")
    text = re.sub(r"[॥।०-९0-9\-]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ======================================================
# LOAD ALL SHLOKAS
# ======================================================
@st.cache_data
def load_all_verses():
    verses = []
    for file in os.listdir(DATA_DIR):
        if file.endswith(".json"):
            with open(os.path.join(DATA_DIR, file), "r", encoding="utf-8") as f:
                data = json.load(f)
                verses.extend(data if isinstance(data, list) else [data])
    return verses

ALL_VERSES = load_all_verses()

# ======================================================
# RETRIEVE SHLOKA (DO NOT CHANGE)
# ======================================================
def retrieve_verse(user_shloka: str):
    user_norm = normalize(user_shloka)
    for verse in ALL_VERSES:
        verse_norm = normalize(verse["sanskrit"]["text"])
        if user_norm == verse_norm:
            return verse
    return None

# ======================================================
# TRANSLATION FUNCTION
# ======================================================
def get_text_in_language(verse, language):
    if language == "English":
        return verse["translations"]["english"]["text"]

    if language == "Hindi":
        return verse["translations"]["hindi"]["text"]

    # Translate from English for other languages
    english_text = verse["translations"]["english"]["text"]
    lang_code = LANGUAGE_CODES[language]
    return translate(english_text, lang_code)

# ======================================================
# TEXT TO SPEECH
# ======================================================
def generate_audio(text, lang_code):
    tts = gTTS(text=text, lang=lang_code)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        return fp.name

# ======================================================
# STREAMLIT UI
# ======================================================
st.set_page_config(page_title="Bhagavad Gita Multilingual Explainer", layout="centered")

st.title("📜 Bhagavad Gita Shloka Explainer (Multilingual + Audio)")

st.markdown(
    "Paste the **complete Sanskrit shloka (Devanāgarī)** below:"
)

user_shloka = st.text_area(
    "Sanskrit Shloka:",
    height=180,
    placeholder="धृतराष्ट्र उवाच ।\nधर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः ।\nमामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय ॥१-१॥"
)

language = st.selectbox(
    "Select Output Language:",
    list(LANGUAGE_CODES.keys())
)

if st.button("Get Meaning + Audio"):
    if not user_shloka.strip():
        st.warning("Please enter a Sanskrit shloka.")
    else:
        verse = retrieve_verse(user_shloka)

        if verse:
            st.success(f"Found: Chapter {verse['chapter']} · Verse {verse['verse']}")

            output_text = get_text_in_language(verse, language)

            st.markdown(f"### 📖 {language} Meaning")
            st.write(output_text)

            audio_path = generate_audio(output_text, LANGUAGE_CODES[language])

            st.markdown("### 🔊 Audio")
            st.audio(audio_path, format="audio/mp3")

        else:
            st.error("Shloka not found in dataset. Please paste the full verse.")
