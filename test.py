import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "facebook/nllb-200-distilled-600M"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

model.to(DEVICE)
model.eval()

def translate_sanskrit_to_english(text):
    # Tokenize with source language
    inputs = tokenizer(
        text,
        return_tensors="pt",
        src_lang="san_Deva"
    ).to(DEVICE)

    # Get target language ID CORRECTLY
    eng_lang_id = tokenizer.convert_tokens_to_ids("eng_Latn")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            forced_bos_token_id=eng_lang_id,
            max_length=128,
            num_beams=5
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)

# ---- TEST ----
shloka = """उद्धरेदात्मनाऽत्मानं नात्मानमवसादयेत् ।
आत्मैव ह्यात्मनो बन्धुरात्मैव रिपुरात्मनः ॥"""
print("Sanskrit:", shloka)
print("English :", translate_sanskrit_to_english(shloka))
