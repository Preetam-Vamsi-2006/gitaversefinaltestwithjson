import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -------------------------------
# CONFIG
# -------------------------------
MODEL_NAME = "ai4bharat/indictrans2-indic-en-1B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_LEN = 128
NUM_BEAMS = 5

# -------------------------------
# LOAD MODEL
# -------------------------------
print("Loading IndicTrans2 model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model.to(DEVICE)
model.eval()

print(f"Model loaded on {DEVICE}")

# -------------------------------
# TRANSLATION FUNCTION
# -------------------------------
def translate_sanskrit_to_english(text: str) -> str:
    """
    Sanskrit → English using IndicTrans2
    """

    # IndicTrans2 expects plain input (no <extra_id>, no prompts)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LEN
    ).to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=MAX_LEN,
            num_beams=NUM_BEAMS,
            early_stopping=True
        )

    return tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    )

# -------------------------------
# INTERACTIVE TEST
# -------------------------------
print("\nEnter Sanskrit text (type 'exit' to quit)\n")

while True:
    shloka = input("🕉️ Sanskrit > ").strip()

    if shloka.lower() in ["exit", "quit"]:
        print("Exiting.")
        break

    if not shloka:
        print("Please enter valid Sanskrit text.")
        continue

    translation = translate_sanskrit_to_english(shloka)

    print("\n📘 English Translation:")
    print(translation)
    print("-" * 60)
