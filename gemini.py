# -*- coding: utf-8 -*-
import google.generativeai as genai
import os
# ----------------------------
# CONFIGURE GEMINI
# ----------------------------

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")

# ----------------------------
# INPUT SANSKRIT SHLOKA
# ----------------------------
shloka = """कर्मण्येवाधिकारस्ते मा फलेषु कदाचन ।
मा कर्मफलहेतुर्भूर्मा ते सङ्गोऽस्त्वकर्मणि ॥"""

# ----------------------------
# PROMPT (IMPORTANT)
# ----------------------------
prompt = f"""
Translate the following Bhagavad Gita shloka into clear, correct English.

Rules:
- Be faithful to the original meaning
- Do NOT add devotion, rebirth, or salvation
- Do NOT add commentary
- Use simple, neutral English
- Output only the translation

Shloka:
{shloka}
"""

# ----------------------------
# GENERATE MEANING
# ----------------------------
response = model.generate_content(prompt)

english_meaning = response.text.strip()

# ----------------------------
# OUTPUT
# ----------------------------
print("📜 Sanskrit Shloka:\n")
print(shloka)

print("\n🌍 English Meaning (Gemini):\n")
print(english_meaning)
