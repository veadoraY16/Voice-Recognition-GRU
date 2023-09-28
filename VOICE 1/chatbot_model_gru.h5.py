import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
import pygame

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('word.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model_gru.h5')

r = sr.Recognizer()

def get_audio():
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=2)
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio, language='id')
            print(f'Anda mengatakan: {text}')
            return text
        except:
            print('Suara tidak terdengar')
            return ""

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def tts(text):
    mp3_fp = BytesIO()
    tts = gTTS(text, lang='id')
    tts.write_to_fp(mp3_fp)
    return mp3_fp

def speak(text):
    pygame.init()
    pygame.mixer.init()
    sound = tts(text)
    sound.seek(0)
    pygame.mixer.music.load(sound, 'mp3')
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(1)

print('GO!_Bot_is_running!')

while True:
    audio_text = get_audio()
    if audio_text:
        print(audio_text)
        ints = predict_class(audio_text)
        res = get_response(ints, intents)
        print(res)
        speak(res)
