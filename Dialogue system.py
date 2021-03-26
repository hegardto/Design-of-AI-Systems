# -*- coding: utf-8 -*-

# -- Sheet --

#Imports
import requests
import json
import urllib
import random
import datetime

#Base URL used to fetch Google Maps data using an API
loc_url= "https://maps.googleapis.com/maps/api/geocode/json?"

#API keys for fetching both location and weather data using Google Maps and OpenWeatherMap APIs
AUTH_KEY_LOC = "AIzaSyBWwlDMKrCtsr7N6fwZ9dtDTQniPq1ro2A"
AUTH_KEY_WEATHER = "25e7ff0a75b3dcc3aec81778039074a8"

#Data for standardizing the way time and dates are written
days = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31']
months = ['01','02','03','04','05','06','07','08','09','10','11','12']
hours = ['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23']
minutes = ['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45','46','47','48','49','50','51','52','53','54','56','57','58','59']

#A frame for answering questions about the weather
class Weather:
    
    #Keywords triggering extract_context() to create a Weather frame
    key_words = ['weather','sunny','rain','cloudy','windy','hot','cold','temperature']

    #Constructor that initiates an empty form
    def __init__(self):
        self.form = [['date',''],['time',''],['location','']]
        self.weather = 0.0

    #Method extracting information from the base sentence 
    #provided by the user, if relevant for the form
    def fill_form(self, sentence):
        
        if type(sentence) == str:
            words = sentence.split()
        else: words = sentence
        
        for word in words:
            
            #Check if the base sentence involves a date
            if '/' in word and len(word) == 5 and self.form[0][1] == '':
                if word[0]+word[1] in days and word[3]+word[4] in months:
                    self.form[0][1] = word[0] + word[1] + '/' + word[3]+word[4]

            #Check if the base sentence involves a time
            if ':' in word and len(word) == 5 and self.form[1][1] == '':
                if word[0]+word[1] in hours and word[3]+word[4] in minutes:
                    self.form[1][1] = word[0] + word[1] + ':' + word[3]+word[4]
            
            #Check if the base sentence involves a location, and then fetch the weather for that location
            parameters = {"address": word, "key": AUTH_KEY_LOC}
            r = requests.get(f"{loc_url}{urllib.parse.urlencode(parameters)}")
            data = json.loads(r.content)
            if data.get("results") and self.form[2][1] == '' and word not in self.key_words:
                self.form[2][1] = data.get("results")[0].get("formatted_address")
                lat = data.get("results")[0].get("geometry").get("location").get("lat")
                lon = data.get("results")[0].get("geometry").get("location").get("lng")
                response = requests.get("https://api.openweathermap.org/data/2.5/onecall?lat=%s&lon=%s&appid=%s&units=metric" % (lat, lon, AUTH_KEY_WEATHER))
                data2 = json.loads(response.text)
                self.weather = data2["current"]["temp"]
    
    #Method for completing the form by asking the user questions
    def check_form(self):

        #Check if date is filled, else ask relevant questions
        if self.form[0][1] == '':   
            word = input("I understood you would like to know the weather, but which date on the format DD/MM?").lower()
            if '/' in word and len(word) == 5:
                if word[0]+word[1] in days and word[3]+word[4] in months:
                    self.form[0][1] = word[0] + word[1] + '/' + word[3]+word[4]
            else:
                while self.form[0][1] == '':
                    word = input("I did not quite understand, could you specify the date on the format DD/MM?").lower()
                    if '/' in word and len(word) == 5:
                        if word[0]+word[1] in days and word[3]+word[4] in months:
                            self.form[0][1] = word[0] + word[1] + '/' + word[3]+word[4]
        
        #Check if time is filled, else ask relevant questions
        if self.form[1][1] == '':   
            word = input("I understood you would like to know the weather on " + self.form[0][1] + " but what time, on the format HH:MM?").lower()
            if ':' in word and len(word) == 5:
                if word[0]+word[1] in hours and word[3]+word[4] in minutes:
                    self.form[1][1] = word[0] + word[1] + ':' + word[3]+word[4]
            else:
                while self.form[1][1] == '':
                    word = input("I did not quite understand, could you specify the time on the format HH:MM?").lower()
                    if ':' in word and len(word) == 5:
                        if word[0]+word[1] in hours and word[3]+word[4] in minutes:
                            self.form[1][1] = word[0] + word[1] + ':' + word[3]+word[4]
    
        #Check if location is filled, else ask relevant questions and fetch weather information for the final location
        if self.form[2][1] != '':
            word = input("I understood you would like to know the weather on the location '" + self.form[2][1] + "'. Is that correct? (Yes/No)").lower()
        if 'no' in word.split():
            self.form[2][1] = ''
        if self.form[2][1] == '':
            word = input("I understood you would like to know the weather on " + self.form[0][1] + " at " + self.form[1][1] + ", but what location?").lower()
            parameters = {"address": word, "key": AUTH_KEY_LOC}
            r = requests.get(f"{loc_url}{urllib.parse.urlencode(parameters)}")
            data = json.loads(r.content)
            if data.get("results") and self.form[2][1] == '':
                self.form[2][1] = data.get("results")[0].get("formatted_address")
                lat = data.get("results")[0].get("geometry").get("location").get("lat")
                lon = data.get("results")[0].get("geometry").get("location").get("lng")
                response = requests.get("https://api.openweathermap.org/data/2.5/onecall?lat=%s&lon=%s&appid=%s&units=metric" % (lat, lon, AUTH_KEY_WEATHER))
                data2 = json.loads(response.text)
                weather = data2["current"]["temp"]

            else: 
                while self.form[2][1] == '':
                    word = input("I did not quite understand, could you specify the location you would like to see the weather for?").lower()
                    parameters = {"address": word, "key": AUTH_KEY_LOC}
                    r = requests.get(f"{loc_url}{urllib.parse.urlencode(parameters)}")
                    data = json.loads(r.content)
                    if data.get("results") and self.form[2][1] == '':
                        self.form[2][1] = data.get("results")[0].get("formatted_address")
                        lat = data.get("results")[0].get("geometry").get("location").get("lat")
                        lon = data.get("results")[0].get("geometry").get("location").get("lng")
                        response = requests.get("https://api.openweathermap.org/data/2.5/onecall?lat=%s&lon=%s&appid=%s&units=metric" % (lat, lon, AUTH_KEY_WEATHER))
                        data2 = json.loads(response.text)
                        weather = data2["current"]["temp"]

        #Put together and print the answer to the user's question using the completed form     
        print("The weather in " + self.form[2][1] + " at " + self.form[0][1] + " " + self.form[1][1] + " is " + str(self.weather) + " degrees celsius.")

#A frame for answering questions about restaurants
class Restaurant:

    #Keywords triggering extract_context() to create a Restaurant frame
    key_words = ['restaurant','food','hungry', 'dish', 'sushi', 'hamburger', 'pizza']
    restaurantdata = ['le chef', 'mcdonalds', 'burger king', 'dunkin donuts']
    price = ['budget','casual','expensive']
    style = ['italian','american','swedish','chinese']

    #Constructor that initiates an empty form
    def __init__(self):
        self.form = [['style',''],['price',''],['location','']]

    #Method extracting information from the base sentence 
    #provided by the user, if relevant for the form
    def fill_form(self, sentence):
        
        if type(sentence) == str:
            words = sentence.split()
        else: words = sentence
        
        for word in words:

            #Check if the base sentence involves a resturant style
            if word in self.style and self.form[0][1] == '':
                self.form[0][1] = word

            #Check if the base sentence involves a budget type
            if word in self.price and self.form[1][1] == '': 
                self.form[1][1] = word
            
            #Check if the base sentence involves a location
            parameters = {"address": word, "key": AUTH_KEY_LOC}
            r = requests.get(f"{loc_url}{urllib.parse.urlencode(parameters)}")
            data = json.loads(r.content)
            if data.get("results") and self.form[2][1] == '' and word not in self.key_words:
                self.form[2][1] = data.get("results")[0].get("formatted_address")

    #Method for completing the form by asking the user questions
    def check_form(self):
    
        #Check if style is filled, else ask relevant questions
        if self.form[0][1] == '':   
            word = input("I understood you would like to go to a restaurant, but what type of restaurant?").lower()
            if type(word) == str:
                words = word.split()
            else: words = word
            for word in words:
                if word in self.style:
                    self.form[0][1] = word
            else:
                while self.form[0][1] == '':
                    word = input("I did not quite understand, could you specify for example if it is a american or italian restaurant?").lower()
                    if type(word) == str:
                        words = word.split()
                    else: words = word
                    for word in words:
                        if word in self.style:
                            self.form[0][1] = word
        
        #Check if price is filled, else ask relevant questions
        if self.form[1][1] == '':   
            word = input("I understood you would like to find a " + self.form[0][1] + " restaurant. But is it a budget/casual/expensive restaurant that you are looking for?").lower()
            if type(word) == str:
                words = word.split()
            else: words = word
            for word in words:
                if word in self.price:
                    self.form[1][1] = word
            else:
                while self.form[1][1] == '':
                    word = input("I did not quite understand, could you specify if it is a budget/casual/expensive restaurant?").lower()
                    if type(word) == str:
                        words = word.split()
                    else: words = word
                    for word in words:
                        if word in self.price:
                            self.form[1][1] = word

        #Check if location is filled, else ask relevant questions
        if self.form[2][1] != '':
            word = input("I understood you would like to find a " + self.form [1][1] + " " + self.form[0][1]+ " restaurant  in " + self.form[2][1] + ". Is that correct? (Yes/No)").lower()
        if 'no' in word.split():
            self.form[2][1] = ''
        if self.form[2][1] == '':
            word = input("I understood you would like to find a " + self.form[0][1] + " restaurant in the price range " + self.form[1][1] + ", but on what location?").lower()
            parameters = {"address": word, "key": AUTH_KEY_LOC}
            r = requests.get(f"{loc_url}{urllib.parse.urlencode(parameters)}")
            data = json.loads(r.content)
            if data.get("results"):
                self.form[2][1] = data.get("results")[0].get("formatted_address")
            else: 
                while self.form[2][1] == '':
                    word = input("I did not quite understand, could you specify the location of the restaurant?")
                    parameters = {"address": word, "key": AUTH_KEY_LOC}
                    r = requests.get(f"{loc_url}{urllib.parse.urlencode(parameters)}")
                    data = json.loads(r.content)
                    if data.get("results"):
                        self.form[2][1] = data.get("results")[0].get("formatted_address")
    
        #Put together and print the answer to the user's question using the completed form       
        print("There is a resturant called " + random.choice(self.restaurantdata)+ " near " + self.form[2][1] + " serving " + self.form[0][1] + " food. The price range is " + self.form[1][1] + ".")

class Bus:
    current_time = datetime.datetime.now()
    key_words = ['bus','tram','bus stop','travel','train', 'transport']
    form = [['from',''],['to',''], ['hour', int(current_time.hour+1)],['minute',int(current_time.minute)], ['correctBusHour', ''], ['correctBusMinute','']]
    busdata = [['gothenburg','kungsbacka', int(11), int(10)], ['gothenburg','kungsbacka',int(13),int(50)], ['gothenburg','kungsbacka',int(10),int(50)]]
    stops = ['gothenburg','kungsbacka']

    #Method extracting information from the base sentence 
    #provided by the user, if relevant for the form
    def fill_form(self, sentence):
        
        if type(sentence) == str:
            words = sentence.split()
        else: words = sentence
        count = 0
        
        for i in range(1,len(words)):
            if words[i] in self.stops and words[i-1] == 'from':
                self.form[0][1] = words[i]
            if words[i] in self.stops and words[i-1] ==  'to':
                self.form[1][1] = words[i]

    # Check if form is filled, otherwise, ask for more information
    def check_form(self):
        # Check if date is filled
        if self.form[0][1] == '':   
            word = input("I understood you would like to travel, but from where?").lower()
            if word in self.stops:
                self.form[0][1] = word
            else:
                while self.form[0][1] == '':
                    word = input("I did not quite understand, could you specify the location that you want to travel from").lower()
                    if word in self.stops:
                        self.form[0][1] = word
        
        # Check if time is filled
        if self.form[1][1] == '':   
            word = input("I understood you would like to travel from " + self.form[0][1] + " but to where?").lower()
            if word in self.stops:
                self.form[1][1] = word
            else:
                while self.form[1][1] == '':
                    word = input("I did not quite understand, could you specify to where?")
                    if word in self.stops:
                        self.form[1][1] = word

        # Check database for buses
        minimum = 60
        minimumHour = 24
        minimumMinute = 60
        for i in range(0,len(self.busdata)):
            #Check if from and to are the same as the entries in the database
            if self.form[0][1] == self.busdata[i][0] and self.form[1][1] == self.busdata[i][1]:
                # Check the bus hour
                if self.form[2][1] == self.busdata[i][2]:
                    #Check if the minute of a bus is "Smaller" than the current next bus
                    if self.busdata[i][3] > self.form [3][1]:
                        if (self.busdata[i][3] - self.form[3][1]) < minimum:
                            self.form[4][1] = self.busdata[i][2]
                            self.form[5][1] = self.busdata[i][3]
                            minimum = (self.busdata[i][3] - self.form[3][1])
            if i == (len(self.busdata)-1) and self.form[4][1] != '':
                print("There is a bus from " + self.form[0][1] + " to " + self.form[1][1] + " leaving at " + str(self.form[4][1]) + ':' + str(self.form[5][1]))
                return

            elif self.form[2][1] < self.busdata[i][2]:
                # Check if the bus hour truly is the earliest
                if (self.busdata[i][2] - self.form[2][1] < minimumHour):        
                    minimumHour = (self.busdata[i][2] - self.form[2][1])
                    self.form[4][1] = self.busdata[i][2]
                    self.form[5][1] = self.busdata[i][3]
                    minimumMinute = self.busdata[i][3]

                elif (self.busdata[i][2] - self.form[2][1] == minimumHour):
                    # Check if the bus minute truly is the earliest
                    if (self.busdata[i][3] < minimumMinute):
                        self.form[5][1] = self.busdata[i][3]
            
            # If there is bus today, print the time
            if i == (len(self.busdata)-1) and self.form[4][1] != '':
                print("There is a bus from " + self.form[0][1] + " to " + self.form[1][1] + " leaving at " + str(self.form[4][1]) + ':' + str(self.form[5][1]))
                return
                
        # If there is no bus today, check the earliest tomorrow        
        else:
            minimumHour2 = 24
            minimumMinute2 = 60
            for i in range(0,len(self.busdata)):
                #Om den går den tidigaste timman
                if self.busdata[i][2] < minimumHour2:
                    minimumHour2 = self.busdata[i][2]
                    minimumMinute2 = self.busdata[i][3]
                    self.form[4][1] = self.busdata[i][2]
                    self.form[5][1] = self.busdata[i][3]
                elif self.busdata[i][2] == minimumHour2:
                        if (self.busdata[i][3] < minimumMinute2):
                            self.form[5][1] = self.busdata[i][3]
                    
            #Print the earliest bus leavning tomorrow
            if i == (len(self.busdata)-1) and self.form[4][1] != '':
                print("Tomorrow there is a bus from " + self.form[0][1] + " to " + self.form[1][1] + " leaving at " + str(self.form[4][1]) + ':' + str(self.form[5][1]))
                return

#Method to extract context from intitial question
def extract_context(sentence):
    context = ''
    while context == '':
        if type(sentence) == str:
                words = sentence.split()
        else: words = sentence

        #Iterate through the words in the initial question and check with
        #domain keywords to see if there is a match between the sentence and domain
        for word in words:
            if word in Weather.key_words:
                context = Weather()

            if word in Restaurant.key_words:
                context = Restaurant()
                
            if word in Bus.key_words: 
                context = Bus()

        if context == '':
            sentence = input("I'm sorry but I don't understand. Could you please ellaborate?")
    return context, sentence

#Main chatbot method
def get_help():
    inputString = input("Hello. I'm David. Can I help you?").lower()
    while 'no' not in inputString.split():
        inputString = input("What can I help you with?").lower()
        context, inputString = extract_context(inputString)
        context.fill_form(inputString)
        context.check_form()
        inputString = input("Do you need any more help?").lower()
    print("Goodbye. Have a nice day!")

get_help()

