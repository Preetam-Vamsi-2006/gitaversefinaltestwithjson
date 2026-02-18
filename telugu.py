from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

shloka = """नारायणं नमस्कृत्य नरं चैव नरोत्तमम् ।
देवीं सरस्वतीं चैव ततो जयमुदीरयेत् ॥"""


inputs = tokenizer(shloka, return_tensors="pt")

# 🔁 CHANGE THIS LINE ONLY
lang_token_id = tokenizer.convert_tokens_to_ids("tel_Telu")  # Telugu
# lang_token_id = tokenizer.convert_tokens_to_ids("hin_Deva")  # Hindi

outputs = model.generate(
    **inputs,
    forced_bos_token_id=lang_token_id,
    max_length=128
)

meaning = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Meaning:")
print(meaning)
