import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
MODEL_NAME = "google/flan-t5-base"   # instruction-tuned (IMPORTANT)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
print("Loading FLAN-T5 model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

model.to(DEVICE)
model.eval()

print(f"Model loaded on {DEVICE}")

# --------------------------------------------------
# FUNCTION: Explain Shloka
# --------------------------------------------------
def explain_shloka(sanskrit_shloka: str, english_meaning: str) -> str:
    """
    Generates a simple explanation given a shloka and its meaning
    """

    prompt = (
        "Explain the following Bhagavad Gita teaching in simple English.\n\n"
        f"Meaning:\n{english_meaning}\n\n"
        "Explanation:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    ).to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=120,
            num_beams=4,
            temperature=0.7,
            no_repeat_ngram_size=3,
            early_stopping=True
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# --------------------------------------------------
# TEST WITH ONE SHLOKA
# --------------------------------------------------
sanskrit = (
    "यदा यदा हि धर्मस्य ग्लानिर्भवति भारत । "
    "अभ्युत्थानमधर्मस्य तदात्मानं सृजाम्यहम् ॥"
)

meaning = (
    "Whenever there is a decline in righteousness and a rise of unrighteousness, "
    "O Arjuna, at that time I manifest Myself."
)

print("\nSanskrit Shloka:")
print(sanskrit)

print("\nGiven Meaning:")
print(meaning)

print("\nGenerated Explanation:")
print(explain_shloka(sanskrit, meaning))
