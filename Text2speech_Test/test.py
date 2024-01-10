#! /usr/bin/python3.5

# ----------------------------------

# https://www.appsloveworld.com/python/289/python-gtts-is-there-a-way-to-change-the-speed-of-the-speech

# Need to insert in terminal:

# $ sudo apt-get update
# $ sudo apt-get install sox
# $ sudo apt-get install libsox-fmt-all

# ----------------------------------

from gtts import gTTS
import os 

file_name = 'Text2speech_Test/test'

mytext='There are a lot of objects in this scene. The mug is red and 8 mm tall'

language = 'en'

tts = gTTS(text=mytext, lang=language, slow=False)

# # save the audio file
tts.save(file_name + '.mp3')

os.system("play " + file_name + ".mp3"+" tempo 1.2")

