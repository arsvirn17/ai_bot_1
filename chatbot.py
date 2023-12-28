import random
import json
import pickle
import numpy as np
from pywebio.output import *
from pywebio import start_server
from pywebio.input import *

import nltk
from nltk.stem import WordNetLemmatizer
from translate import Translator

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5')

def main():
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

    put_text("Go! Bot is running!")


    msg_box = output()
    put_scrollable(msg_box, height=300, keep_bottom=True)

    translator = Translator(from_lang="ru", to_lang="en")

    while True: 
        global chat_msgs
        message = input("")
        msg_box.append(put_markdown(message))
        ints = predict_class(message)
        res = get_response(ints, intents)
        msg_box.append(put_markdown(res))

if __name__ == "__main__":
       start_server(main, debug=True, port=8080, cdn=False)