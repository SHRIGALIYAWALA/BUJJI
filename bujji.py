import cv2
import numpy as np
import pyttsx3
import random
import os
import datetime
import requests
from googletrans import Translator
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time
import feedparser
import speech_recognition as sr
import openai
from forex_python.converter import CurrencyRates
from bs4 import BeautifulSoup
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle
import shutil
import webbrowser


openai.api_key = "sk-proj-llEsR2HSu8BwsZXA3uTXBoTrcJAH7ekOJDMnnMH7AnBEqdC-A2uNn9mLCnwX3FXVLgY0Kl31HBT3BlbkFJL2icBMq0SVYIXjud06bmJpwDs_qMDKo0VGaZ3uI-_u1XEh5vgGMXEnIX5GJDuxx5EpBtMhAA"

# Speech Command for interacting with Bujji
class SpeechCommand:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        try:
            self.engine = pyttsx3.init(driverName='sapi5')
        except Exception as e:
            print(f"Error initializing pyttsx3: {e}")

    def speak(self, text):
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"Error during speech synthesis: {e}")

    def listen(self, retries=3, timeout=5):
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source)
            print("Listening...")
            audio = None
            while retries > 0:
                try:
                    audio = self.recognizer.listen(source, timeout=timeout)
                    break
                except sr.WaitTimeoutError:
                    print("Timeout, retrying...")
                    retries -= 1
                except Exception as e:
                    print(f"Error in speech recognition: {e}")
                    retries -= 1
            if audio:
                try:
                    command = self.recognizer.recognize_google(audio)
                    print(f"Recognized: {command}")
                    return command
                except sr.UnknownValueError:
                    print("Could not understand audio")
                except Exception as e:
                    print(f"Error in speech recognition: {e}")
            return None

    def translate_text(self, text, target_language):

    translator = Translator()
    translation = translator.translate(text, dest=target_language)
    return translation.text

    def chatgpt_response(prompt):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response['choices'][0]['message']['content']

    def set_timer(seconds):
        print(f"Timer set for {seconds} seconds.")
        time.sleep(seconds)
        print("Time's up!")

    def set_reminder(message, reminder_time):
        print(f"Reminder set for {reminder_time}.")
        while True:
            if datetime.datetime.now() >= reminder_time:
                print(f"Reminder: {message}")
                break
            time.sleep(1)

    def organize_files(directory):
        if not os.path.exists(directory):
            print("Directory does not exist.")
            return

        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path):
                file_extension = file.split('.')[-1]
                folder = os.path.join(directory, file_extension)
                if not os.path.exists(folder):
                    os.makedirs(folder)
                shutil.move(file_path, os.path.join(folder, file))
        print("Files organized.")

    def system_control(action):
        if action == "shutdown":
            os.system("shutdown /s /t 1")
        elif action == "restart":
            os.system("shutdown /r /t 1")
        elif action == "logout":
            os.system("shutdown -l")
        else:
            print("Invalid action.")

    def search_web(query):
        url = f"https://www.google.com/search?q={query}"
        webbrowser.open(url)
        print(f"Searching for {query}...")
# Hotword Detector
class HotwordDetector:
    def __init__(self, hotword="bujji"):
        self.hotword = hotword.lower()
        self.recognizer = sr.Recognizer()

    def start_detection(self):
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source)
            while True:
                print("Waiting for hotword...")
                try:
                    audio = self.recognizer.listen(source)
                    command = self.recognizer.recognize_google(audio).lower()
                    if self.hotword in command:
                        print("Hotword detected!")
                        return True
                except sr.UnknownValueError:
                    pass
                except Exception as e:
                    print(f"Error in hotword detection: {e}")
# News Fetcher
class NewsFetcher:
    def __init__(self):
        self.rss_feed = "https://news.google.com/news/rss"

    def fetch_news(self):
        try:
            news = feedparser.parse(self.rss_feed)
            headlines = []
            for entry in news.entries[:5]:
                headlines.append(entry.title)
            return "\n".join(headlines)
        except Exception as e:
            print(f"Error fetching news: {e}")
        return "Unable to fetch news."

# Joke Generator
class JokeGenerator:
    def fetch_joke(self):
        try:
            jokes = [
                "Why don’t skeletons fight each other? They don’t have the guts.",
                "I told my computer I needed a break, and now it won’t stop sending me ads for vacation destinations.",
                "Why don’t eggs tell jokes? They’d crack each other up.",
                "I used to play piano by ear, but now I use my hands."
            ]
            return random.choice(jokes)
        except Exception as e:
            print(f"Error fetching joke: {e}")
        return "Unable to fetch joke."

# To-Do List Handler
class TodoListHandler:
    def __init__(self):
        self.todos = []

    def add_task(self, task):
        self.todos.append(task)
        return f"Task '{task}' added to your to-do list."

    def remove_task(self, task):
        if task in self.todos:
            self.todos.remove(task)
            return f"Task '{task}' removed from your to-do list."
        return f"Task '{task}' not found."

    def view_tasks(self):
        if self.todos:
            return "\n".join(self.todos)
        return "Your to-do list is empty."

# Weather Fetcher
class WeatherFetcher:
    def __init__(self):
        self.api_key = '3afbda184a4db24230908727aaf1cdf6'

    def fetch_weather(self, location):
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={self.api_key}&units=metric"
            response = requests.get(url)
            data = response.json()
            if data['cod'] == 200:
                weather = data['weather'][0]['description']
                temperature = data['main']['temp']
                return f"The current weather in {location} is {weather} with a temperature of {temperature}°C."
            else:
                return "Sorry, I couldn't get the weather information."
        except Exception as e:
            print(f"Error fetching weather: {e}")
        return "Unable to fetch weather."

# Currency Converter
class CurrencyConverter:
    def __init__(self):
        self.currency_rate = CurrencyRates()

    def convert(self, from_currency, to_currency, amount):
        try:
            converted_amount = self.currency_rate.convert(from_currency, to_currency, amount)
            return f"{amount} {from_currency} is equivalent to {converted_amount} {to_currency}."
        except Exception as e:
            print(f"Error converting currency: {e}")
        return "Unable to convert currency."

# Stock Price Checker
class StockPriceChecker:
    def fetch_stock_price(self, symbol):
        try:
            url = f"https://finance.yahoo.com/quote/{symbol}"
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            price = soup.find('span', {'data-reactid': '50'}).text
            return f"The current price of {symbol} is {price}."
        except Exception as e:
            print(f"Error fetching stock price: {e}")
        return "Unable to fetch stock price."

# Dictionary Lookup
class DictionaryLookup:
    def lookup_word(self, word):
        try:
            url = f"https://www.dictionary.com/browse/{word}"
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            definition = soup.find('span', {'class': 'one-click-content'}).text
            return f"The definition of {word} is: {definition}"
        except Exception as e:
            print(f"Error looking up word: {e}")
        return "Unable to find word definition."

# Object Detection (Placeholder with OpenCV)
class ObjectDetection:
    def __init__(self):
        self.net = cv2.dnn.readNet(r"C:\Users\shrig\bujji\assets\yolov3.weights", r"C:\Users\shrig\bujji\assets\yolov3.cfg")
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

    def detect_objects(self, frame):
        height, width, channels = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        class_ids, confidences, boxes = [], [], []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = center_x - w // 2
                    y = center_y - h // 2
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        return f"Detected {len(indexes)} objects."

# Emotion Detection (Placeholder)
class EmotionDetection:
    def __init__(self):
        # Load the model without compiling to avoid issues with the deprecated 'lr' argument
        self.model = tf.keras.models.load_model(
            r"C:\Users\shrig\bujji\assets\emotion_model.hdf5", compile=False
        )

        # Recompile the model with the correct optimizer configuration
        self.model.compile(optimizer=Adam(learning_rate=0.0001), 
                           loss='categorical_crossentropy', 
                           metrics=['accuracy'])

    def detect_emotion(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + r"haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (48, 48))
            face_roi = face_roi.reshape(1, 48, 48, 1)
            face_roi = face_roi.astype('float32') / 255.0

            emotion = self.model.predict(face_roi)
            emotion_label = np.argmax(emotion)
            return f"Detected emotion: {emotion_label}"
        return "No face detected."

# Health Detection (Placeholder)
class HealthDetection:
    def detect_health(self, frame):
        return "Health detection complete. No specific functionality yet implemented."
# Spotify Integration
class MusicController:
    def __init__(self):
        self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id="330ba67232f64fecbe54cdad356ef9d8",
                                                             client_secret="3e1c8f6ec4fa4f05ba27702a6017085c",
                                                             redirect_uri="http://localhost:8888/callback",
                                                             scope="user-read-playback-state,user-modify-playback-state"))
    
    def play_song(self, song_name):
        results = self.sp.search(q=song_name, limit=1)
        if results['tracks']['items']:
            track = results['tracks']['items'][0]
            self.sp.start_playback(uris=[track['uri']])
            return f"Playing {track['name']} by {track['artists'][0]['name']}."
        return "Song not found."

    def pause_song(self):
        self.sp.pause_playback()
        return "Playback paused."

    def next_song(self):
        self.sp.next_track()
        return "Skipping to the next track."

# Google Calendar Integration
class CalendarController:
    def __init__(self):
        self.SCOPES = ['https://www.googleapis.com/auth/calendar']
        self.service = self.authenticate_google()

    def authenticate_google(self):
        creds = None
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(r"C:\Users\shrig\bujji\assets\client_secret_307090682704-nfep9p3pjrdvhepvducilo2fc5h33boj.apps.googleusercontent.com.json", self.SCOPES)
                creds = flow.run_local_server(port=0)
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)
        return build('calendar', 'v3', credentials=creds)

    def create_event(self, summary, start_time, end_time):
        event = {
            'summary': summary,
            'start': {
                'dateTime': start_time,
                'timeZone': 'UTC',
            },
            'end': {
                'dateTime': end_time,
                'timeZone': 'UTC',
            }
        }
        event = self.service.events().insert(calendarId='primary', body=event).execute()
        return f"Event '{summary}' created."

    def list_events(self):
        now = datetime.datetime.utcnow().isoformat() + 'Z'
        events_result = self.service.events().list(calendarId='primary', timeMin=now,
                                                   maxResults=10, singleEvents=True,
                                                   orderBy='startTime').execute()
        events = events_result.get('items', [])
        if not events:
            return 'No upcoming events found.'
        return "\n".join([f"{event['summary']} at {event['start']['dateTime']}" for event in events])

import numpy as np


# Main Controller with feature selection
class MainController:
    def __init__(self):
        self.speech_command = SpeechCommand()
        self.news_fetcher = NewsFetcher()
        self.joke_generator = JokeGenerator()
        self.todo_list_handler = TodoListHandler()
        self.weather_fetcher = WeatherFetcher()
        self.currency_converter = CurrencyConverter()
        self.stock_price_checker = StockPriceChecker()
        self.dictionary_lookup = DictionaryLookup()
        self.object_detection = ObjectDetection()
        self.emotion_detection = EmotionDetection()
        self.health_detection = HealthDetection()
        self.hotword_detector = HotwordDetector()
        self.music_controller = MusicController()
        self.calendar_controller = CalendarController()
        
    
                
    def run(self):
        print("Starting Main Controller")
        self.speech_command.speak("Hello! I am Bujji. I can help you with various tasks. Please tell me what you'd like to do.")
        self.ask_for_action()

    def ask_for_action(self):
        self.speech_command.speak(
            "What would you like me to do? You can say: News, Joke, To-Do, Weather, Currency, Stock, Dictionary, Emotion, Object, Health, Music, or Calendar."
        )
        
        user_command = self.speech_command.listen().lower()
        action_map = {
            "news": self.run_news,
            "joke": self.run_joke,
            "to-do": self.run_todo,
            "weather": self.run_weather,
            "currency": self.run_currency_converter,
            "stock": self.run_stock_price_checker,
            "dictionary": self.run_dictionary_lookup,
            "emotion": self.run_emotion_detection,
            "object": self.run_object_detection,
            "health": self.run_health_detection,
            "play music": self.run_play_music,
            "pause music": self.run_pause_music,
            "next song": self.run_next_song,
            "calendar": self.run_calendar,

            "translate": self.handle_translate,
            "chat": self.handle_chatgpt,
            "timer": self.handle_timer,
            "reminder": self.handle_reminder,
            "organize": self.handle_organize_files,
            "control": self.handle_system_control,
            "search": self.handle_search_web,
        }

        for key, action in action_map.items():
            if key in user_command:
                action()
                break
        else:
            self.speech_command.speak("I didn't understand that command. Please try again.")
            self.ask_for_action()

    def handle_translate(self):
        text = input("Enter text to translate: ")
        language = input("Enter target language (e.g., 'es' for Spanish): ")
        print("Translated text:", self.translate_text(text, language))


    def handle_chatgpt(self):
        prompt = input("Enter your question or prompt: ")
        print("ChatGPT response:", self.chatgpt_response(prompt))


    def handle_timer(self):
        seconds = int(input("Enter timer duration in seconds: "))
        self.set_timer(seconds)


    def handle_reminder(self):
        message = input("Enter reminder message: ")
        time_input = input("Enter reminder time (YYYY-MM-DD HH:MM:SS): ")
        reminder_time = datetime.datetime.strptime(time_input, "%Y-%m-%d %H:%M:%S")
        self.set_reminder(message, reminder_time)


    def handle_organize_files(self):
        directory = input("Enter directory path to organize: ")
        self.organize_files(directory)


    def handle_system_control(self):
        action = input("Enter action (shutdown/restart/logout): ")
        self.system_control(action)


    def handle_search_web(self):
        query = input("Enter search query: ")
        self.search_web(query)

        
    def run_news(self):
        self.speech_command.speak("Fetching news...")
        news = self.news_fetcher.fetch_news()
        self.speech_command.speak(news)

    def run_joke(self):
        self.speech_command.speak("Fetching a joke...")
        joke = self.joke_generator.fetch_joke()
        self.speech_command.speak(joke)

    def run_todo(self):
        self.speech_command.speak("You can add, remove, or view tasks in your to-do list.")
        command = self.speech_command.listen()
        self.handle_todo_command(command)

    def handle_todo_command(self, command):
        if "add" in command:
            task = command.replace("add", "").strip()
            response = self.todo_list_handler.add_task(task)
        elif "remove" in command:
            task = command.replace("remove", "").strip()
            response = self.todo_list_handler.remove_task(task)
        elif "view" in command:
            response = self.todo_list_handler.view_tasks()
        else:
            response = "I didn't understand that to-do command."
        self.speech_command.speak(response)

    def run_weather(self):
        self.speech_command.speak("Which location would you like the weather for?")
        location = self.speech_command.listen()
        self.fetch_weather_for_location(location)

    def fetch_weather_for_location(self, location):
        if location:
            weather = self.weather_fetcher.fetch_weather(location)
            self.speech_command.speak(weather)
        else:
            self.speech_command.speak("I couldn't get the location.")

    def run_currency_converter(self):
        self.speech_command.speak("Tell me the amount, from currency, and to currency.")
        details = self.speech_command.listen()
        self.convert_currency(details)

    def convert_currency(self, details):
        if details:
            try:
                amount, from_currency, to_currency = details.split()
                amount = float(amount)
                conversion = self.currency_converter.convert(from_currency.upper(), to_currency.upper(), amount)
                self.speech_command.speak(conversion)
            except ValueError:
                self.speech_command.speak("I couldn't understand the currency conversion details.")
        else:
            self.speech_command.speak("I couldn't get the currency conversion details.")

    def run_stock_price_checker(self):
        self.speech_command.speak("Which company's stock price would you like?")
        symbol = self.speech_command.listen()
        self.fetch_stock_price(symbol)

    def fetch_stock_price(self, symbol):
        if symbol:
            stock_price = self.stock_price_checker.fetch_stock_price(symbol.upper())
            self.speech_command.speak(stock_price)
        else:
            self.speech_command.speak("I couldn't get the company symbol.")

    def run_dictionary_lookup(self):
        self.speech_command.speak("Which word would you like to look up?")
        word = self.speech_command.listen()
        self.lookup_word(word)

    def lookup_word(self, word):
        if word:
            definition = self.dictionary_lookup.lookup_word(word)
            self.speech_command.speak(definition)
        else:
            self.speech_command.speak("I couldn't get the word.")

    def run_emotion_detection(self):
        self.speech_command.speak("Analyzing emotions...")
        frame = np.zeros((480, 640, 3), np.uint8)  # Dummy frame
        emotion = self.emotion_detection.detect_emotion(frame)
        self.speech_command.speak(emotion)

    def run_object_detection(self):
        self.speech_command.speak("Analyzing objects...")
        frame = np.zeros((480, 640, 3), np.uint8)  # Dummy frame
        objects_detected = self.object_detection.detect_objects(frame)
        self.speech_command.speak(objects_detected)

    def run_health_detection(self):
        self.speech_command.speak("Analyzing health...")
        frame = np.zeros((480, 640, 3), np.uint8)  # Dummy frame
        health = self.health_detection.detect_health(frame)
        self.speech_command.speak(health)

    def run_play_music(self):
        self.speech_command.speak("Which song would you like to play?")
        song_name = self.speech_command.listen()
        if song_name:
            response = self.music_controller.play_song(song_name)
            self.speech_command.speak(response)
        else:
            self.speech_command.speak("I couldn't hear the song name.")

    def run_pause_music(self):
        response = self.music_controller.pause_song()
        self.speech_command.speak(response)

    def run_next_song(self):
        response = self.music_controller.next_song()
        self.speech_command.speak(response)

    def run_calendar(self):
        self.speech_command.speak("You can say: Create Event or List Events.")
        command = self.speech_command.listen()
        self.handle_calendar_command(command)

    def handle_calendar_command(self, command):
        if "create" in command:
            title, start_time, end_time = self.get_calendar_event_details()
            response = self.calendar_controller.create_event(title, start_time, end_time)
        elif "list" in command:
            response = self.calendar_controller.list_events()
        else:
            response = "I didn't understand that calendar command."
        self.speech_command.speak(response)

    def get_calendar_event_details(self):
        self.speech_command.speak("What is the event title?")
        title = self.speech_command.listen()
        self.speech_command.speak("What is the start time in YYYY-MM-DDTHH:MM:SS format?")
        start_time = self.speech_command.listen()
        self.speech_command.speak("What is the end time in YYYY-MM-DDTHH:MM:SS format?")
        end_time = self.speech_command.listen()
        return title, start_time, end_time

if __name__ == "__main__":
    controller = MainController()
    controller.speech_command.speak("Bujji is now ready. Say 'Bujji' to activate.")
    if controller.hotword_detector.start_detection():
        controller.run()
