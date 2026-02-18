# ------------------------------------------------------------
# Bhagavad Gita Shloka → Meaning (English + Telugu + Hindi)
# Sanskrit → English : NLLB model
# English → Telugu/Hindi : mtranslate
# Python 3.11
# ------------------------------------------------------------
# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from mtranslate import translate

# -----------------------------
# LOAD MODEL (ONCE)
# -----------------------------
MODEL_NAME = "facebook/nllb-200-distilled-600M"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# -----------------------------
# INPUT SHLOKA
# -----------------------------
shloka = """उद्धरेदात्मनाऽत्मानं नात्मानमवसादयेत् ।
आत्मैव ह्यात्मनो बन्धुरात्मैव रिपुरात्मनः ॥"""

# -----------------------------
# STEP 1: SANSKRIT → ENGLISH
# -----------------------------
inputs = tokenizer(shloka, return_tensors="pt")

eng_token_id = tokenizer.convert_tokens_to_ids("eng_Latn")

output_en = model.generate(
    **inputs,
    forced_bos_token_id=eng_token_id,
    max_length=160,
    num_beams=5,
    no_repeat_ngram_size=2,
    early_stopping=True
)

english_meaning = tokenizer.decode(output_en[0], skip_special_tokens=True)

# -----------------------------
# STEP 2: ENGLISH → TELUGU & HINDI
# -----------------------------
telugu_meaning = translate(english_meaning, "te")
hindi_meaning = translate(english_meaning, "hi")

# -----------------------------
# OUTPUT
# -----------------------------
print("\n📜 Sanskrit Shloka:\n")
print(shloka)

print("\n🌍 English Meaning (AI-generated):\n")
print(english_meaning)

print("\n🌿 Telugu Meaning (from English):\n")
print(telugu_meaning)

print("\n🇮🇳 Hindi Meaning (from English):\n")
print(hindi_meaning)
