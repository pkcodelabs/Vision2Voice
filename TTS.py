import pyttsx3
from gtts import gTTS
import playsound

def speak_text(text: str, lang: str = "hi"):
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save("output.mp3")
    playsound.playsound("output.mp3")


def speak_text_offline(text: str, lang: str = "hi"):
    engine = pyttsx3.init()
    voices = engine.getProperty("voices")

    # Try selecting a Hindi voice if installed
    for voice in voices:
        if "hi" in voice.languages or "hindi" in voice.name.lower():
            engine.setProperty("voice", voice.id)
            break

    engine.setProperty("rate", 150)   # speaking speed
    engine.setProperty("volume", 1.0) # max volume
    engine.say(text)
    engine.runAndWait()

# Example usage
if __name__ == "__main__":
    hindi_text = "नमस्ते, यह आपका अनुवादित पाठ है।"
    speak_text_offline(hindi_text)
