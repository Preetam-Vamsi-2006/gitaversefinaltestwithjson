# -*- coding: utf-8 -*-

from gtts import gTTS
from playsound import playsound
import tempfile
import os

# Sanskrit shloka to recite
shloka = """कर्मण्येवाधिकारस्ते मा फलेषु कदाचन ।
मा कर्मफलहेतुर्भूर्मा ते सङ्गोऽस्त्वकर्मणि ॥"""

# Create temporary file
with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
    temp_path = fp.name

# Generate TTS audio (Hindi voice works best for Sanskrit)
tts = gTTS(text=shloka, lang="hi")
tts.save(temp_path)

# Play audio
playsound(temp_path)

# Delete temporary file
os.remove(temp_path)
