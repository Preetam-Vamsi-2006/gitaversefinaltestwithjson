import json
import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = "BhagavatGitaJsonFiles"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Normalize Sanskrit
# -----------------------------
def normalize(text):
    text = text.replace("\n", " ")
    text = re.sub(r"[॥0-9\-]", "", text)
    return re.sub(r"\s+", " ", text).strip()

# -----------------------------
# Load all verses
# -----------------------------
def load_all_verses():
    verses = []
    for file in os.listdir(DATA_DIR):
        if file.endswith(".json"):
            with open(os.path.join(DATA_DIR, file), "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    verses.extend(data)
                else:
                    verses.append(data)
    return verses

ALL_VERSES = load_all_verses()

# -----------------------------
# Retrieve English meaning
# -----------------------------
def retrieve_meaning(user_shloka):
    user_norm = normalize(user_shloka)

    for verse in ALL_VERSES:
        verse_norm = normalize(verse["sanskrit"]["text"])
        if user_norm == verse_norm:
            return verse

    return None

# -----------------------------
# Load FLAN-T5
# -----------------------------
FLAN_MODEL = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(FLAN_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(FLAN_MODEL).to(DEVICE)
model.eval()

# -----------------------------
# FLAN-T5 Explanation
# -----------------------------
def explain_with_flan(meaning):
    prompt = (
        "Explain the following Bhagavad Gita verse meaning "
        "in simple and clear English without adding new ideas:\n\n"
        f"{meaning}"
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
            max_length=120,
            num_beams=4
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    print("Enter Sanskrit Shloka:")
    user_input = input().strip()

    verse = retrieve_meaning(user_input)

    if verse:
        meaning = verse["translations"]["english"]["text"]

        print("\n✅ Retrieved English Meaning:")
        print(meaning)

        print("\n🔹 FLAN-T5 Explanation:")
        print(explain_with_flan(meaning))
    else:
        print("\n❌ Shloka not found in dataset.")
