import streamlit as st
import json
import os
import re
import unicodedata
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ======================================================
# CONFIG
# ======================================================
DATA_DIR = "BhagavatGitaJsonFiles"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FLAN_MODEL = "google/flan-t5-base"

# ======================================================
# NORMALIZE SANSKRIT (CRITICAL)
# ======================================================
def normalize(text: str) -> str:
    # Unicode normalization (handles Sanskrit properly)
    text = unicodedata.normalize("NFKD", text)

    # Convert newlines to spaces
    text = text.replace("\n", " ")

    # Remove danda, double danda, Devanagari digits, normal digits, hyphens
    text = re.sub(r"[॥।०-९0-9\-]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()

# ======================================================
# LOAD ALL SHLOKAS (CACHE)
# ======================================================
@st.cache_data
def load_all_verses():
    verses = []

    for file in os.listdir(DATA_DIR):
        if file.endswith(".json"):
            path = os.path.join(DATA_DIR, file)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

                if isinstance(data, list):
                    verses.extend(data)
                else:
                    verses.append(data)

    return verses

ALL_VERSES = load_all_verses()

# ======================================================
# RETRIEVE MEANING
# ======================================================
def retrieve_verse(user_shloka: str):
    user_norm = normalize(user_shloka)

    for verse in ALL_VERSES:
        verse_text = verse["sanskrit"]["text"]
        verse_norm = normalize(verse_text)

        if user_norm == verse_norm:
            return verse

    return None

# ======================================================
# LOAD FLAN-T5 (CACHE)
# ======================================================
@st.cache_resource
def load_flan():
    tokenizer = AutoTokenizer.from_pretrained(FLAN_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(FLAN_MODEL).to(DEVICE)
    model.eval()
    return tokenizer, model

tokenizer, model = load_flan()

# ======================================================
# FLAN-T5 EXPLANATION (SAFE)
# ======================================================
def explain_with_flan(meaning: str) -> str:
    prompt = (
        "Explain the following Bhagavad Gita verse meaning "
        "in one or two clear sentences. "
        "Do not repeat words unnecessarily.\n\n"
        f"Meaning: {meaning}"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256
    ).to(DEVICE)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=80,                 # 🔹 shorter output
            num_beams=4,
            repetition_penalty=1.8,        # 🔹 prevents loops
            no_repeat_ngram_size=3,         # 🔹 blocks phrase repetition
            early_stopping=True
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)
# ======================================================
# STREAMLIT UI
# ======================================================
st.set_page_config(page_title="Bhagavad Gita Explainer", layout="centered")

st.title("📜 Bhagavad Gita Shloka Explainer")

st.markdown(
    """
Enter the **complete Sanskrit shloka** in **Devanāgarī**  
(multi-line input is supported).
"""
)

user_shloka = st.text_area(
    "Paste Sanskrit Shloka here:",
    height=180,
    placeholder="धृतराष्ट्र उवाच ।\nधर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः ।\nमामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय ॥१-१॥"
)

if st.button("Get Meaning & Explanation"):
    if not user_shloka.strip():
        st.warning("Please enter a Sanskrit shloka.")
    else:
        verse = retrieve_verse(user_shloka)

        if verse:
            st.success(f"Found: Chapter {verse['chapter']} · Verse {verse['verse']}")

            st.markdown("### 📖 English Meaning")
            meaning = verse["translations"]["english"]["text"]
            st.write(meaning)

            st.markdown("### 🔍 Simple Explanation (FLAN-T5)")
            st.write(explain_with_flan(meaning))

        else:
            st.error("Shloka not found in dataset. Please paste the full verse.")
