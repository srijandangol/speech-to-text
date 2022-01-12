# import library

import speech_recognition as sr

# Initialize recognizer class (for recognizing the speech)

def get_text():
    r = sr.Recognizer()

    # Reading Microphone as source
    # listening the speech and store in audio_text variable
    with sr.Microphone() as mic:
        print("Talk")
        audio = r.listen(mic)
        try:
            Audio_text = r.recognize_google(audio)
            print('{}'.format(Audio_text))
        except:
            print('sorry')
    return Audio_text