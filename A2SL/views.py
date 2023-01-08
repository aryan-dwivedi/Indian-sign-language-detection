from django.http import HttpResponse, StreamingHttpResponse
from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import login, logout
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import tensorflow as tf
from tensorflow import keras
import skimage
import numpy as np
import cv2
from PIL import Image

from django.contrib.staticfiles import finders
from django.contrib.auth.decorators import login_required

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

model = keras.models.load_model(r"assets/model.h5")

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()

        # flip the image
        image = cv2.flip(image, 1)

        x1 = int(0.7 * image.shape[1])
        y1 = 5
        x2 = image.shape[1] - 5
        y2 = int(0.3 * image.shape[1])
        cv2.rectangle(image, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1),
                        (255, 255, 255), 3)
                        
        ret, jpeg = cv2.imencode('.jpg', image)
        
        return jpeg.tobytes()
    
    def get_jpg(self):
        success, image = self.video.read()

        # flip the image
        image = cv2.flip(image, 1)

        x1 = int(0.7 * image.shape[1])
        y1 = 5
        x2 = image.shape[1] - 5
        y2 = int(0.3 * image.shape[1])

        # crop the image
        crop_img = image[y1:y2, x1:x2]

        return crop_img


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def home_view(request):
    return render(request, 'home.html')


def about_view(request):
    return render(request, 'about.html')


def text_to_sign(request):
    if request.method == 'POST':
        text = request.POST.get('sen')
        # tokenizing the sentence
        text.lower()
        # tokenizing the sentence
        words = word_tokenize(text)

        tagged = nltk.pos_tag(words)
        tense = {}
        tense["future"] = len([word for word in tagged if word[1] == "MD"])
        tense["present"] = len(
            [word for word in tagged if word[1] in ["VBP", "VBZ", "VBG"]])
        tense["past"] = len(
            [word for word in tagged if word[1] in ["VBD", "VBN"]])
        tense["present_continuous"] = len(
            [word for word in tagged if word[1] in ["VBG"]])

        # stopwords that will be removed
        stop_words = set(["mightn't", 're', 'wasn', 'wouldn', 'be', 'has', 'that', 'does', 'shouldn', 'do', "you've", 'off', 'for', "didn't", 'm', 'ain', 'haven', "weren't", 'are', "she's", "wasn't", 'its', "haven't", "wouldn't", 'don', 'weren', 's', "you'd", "don't", 'doesn', "hadn't", 'is', 'was', "that'll", "should've", 'a', 'then', 'the', 'mustn', 'i',
                         'nor', 'as', "it's", "needn't", 'd', 'am', 'have',  'hasn', 'o', "aren't", "you'll", "couldn't", "you're", "mustn't", 'didn', "doesn't", 'll', 'an', 'hadn', 'whom', 'y', "hasn't", 'itself', 'couldn', 'needn', "shan't", 'isn', 'been', 'such', 'shan', "shouldn't", 'aren', 'being', 'were', 'did', 'ma', 't', 'having', 'mightn', 've', "isn't", "won't"])

        # removing stopwords and applying lemmatizing nlp process to words
        lr = WordNetLemmatizer()
        filtered_text = []
        for w, p in zip(words, tagged):
            if w not in stop_words:
                if p[1] == 'VBG' or p[1] == 'VBD' or p[1] == 'VBZ' or p[1] == 'VBN' or p[1] == 'NN':
                    filtered_text.append(lr.lemmatize(w, pos='v'))
                elif p[1] == 'JJ' or p[1] == 'JJR' or p[1] == 'JJS' or p[1] == 'RBR' or p[1] == 'RBS':
                    filtered_text.append(lr.lemmatize(w, pos='a'))

                else:
                    filtered_text.append(lr.lemmatize(w))

        # adding the specific word to specify tense
        words = filtered_text
        temp = []
        for w in words:
            if w == 'I':
                temp.append('Me')
            else:
                temp.append(w)
        words = temp
        probable_tense = max(tense, key=tense.get)

        if probable_tense == "past" and tense["past"] >= 1:
            temp = ["Before"]
            temp = temp + words
            words = temp
        elif probable_tense == "future" and tense["future"] >= 1:
            if "Will" not in words:
                temp = ["Will"]
                temp = temp + words
                words = temp
            else:
                pass
        elif probable_tense == "present":
            if tense["present_continuous"] >= 1:
                temp = ["Now"]
                temp = temp + words
                words = temp

        filtered_text = []
        for w in words:
            path = w + ".mp4"
            f = finders.find(path)
            # splitting the word if its animation is not present in database
            if not f:
                for c in w:
                    filtered_text.append(c)
            # otherwise animation of word
            else:
                filtered_text.append(w)
        words = filtered_text

        return render(request, 'texttosign.html', {'words': words, 'text': text})
    else:
        return render(request, 'texttosign.html')


def video_feed(request):
    return StreamingHttpResponse(gen(VideoCamera()),
                                 content_type='multipart/x-mixed-replace; boundary=frame')

def subtitles(request):
    img_file = VideoCamera().get_jpg()
    imageSize = 50
    img_file = skimage.transform.resize(img_file, (imageSize, imageSize, 3))
    img_arr = np.asarray(img_file)
    img_arr = tf.expand_dims(img_arr, axis=0)
    y_pred = model.predict(img_arr)
    label = np.argmax(y_pred, axis=1)[0]
    subtitles = 'Error'
    if 26 <= label <= 29:
        if label == 26:
            subtitles = 'del'
        elif label == 27:
            subtitles = 'nothing'
        else:
            subtitles = ' '
    else:
        subtitles = chr(65 + label)

    print(subtitles)
    return HttpResponse(subtitles)

def sign_to_text(request):
    return render(request, 'signtotext.html')
