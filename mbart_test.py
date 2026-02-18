import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

model.to(DEVICE)
model.eval()

def translate_sanskrit_to_english(text):
    tokenizer.src_lang = "sa_IN"

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    ).to(DEVICE)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"],
            num_beams=5,
            max_length=128
        )

    return tokenizer.decode(generated[0], skip_special_tokens=True)

# Test
shloka = "कर्मण्येवाधिकारस्ते मा फलेषु कदाचन"
print("English:", translate_sanskrit_to_english(shloka))
