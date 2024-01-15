#! /usr/bin/python3.5

# ----------------------------------

# https://www.appsloveworld.com/python/289/python-gtts-is-there-a-way-to-change-the-speed-of-the-speech

# Need to insert in terminal:

# $ sudo apt-get update
# $ sudo apt-get install sox
# $ sudo apt-get install libsox-fmt-all

# ----------------------------------

from time import sleep
from gtts import gTTS
import os
import threading


def voice():
    

    file_name = 'Text2speech_Test/test'
    mytext='There are a lot of objects in this scene. The mug is red and 8 mm tall'
    language = 'en'
    tts = gTTS(text=mytext, lang=language, slow=False)

    # save the audio file
    tts.save(file_name + '.mp3')
    os.system("play " + file_name + ".mp3"+" tempo 1.2")
    os.remove(file_name + '.mp3')

for i in range(15):
    print(str(i))
    sleep(1)
    if i==7:
        thread = threading.Thread(target=voice, args=())
        thread.start()


# a = [1,2,3,10,4,9]

# idx_max = a.index(max(a))

# print(idx_max)

# a = [1, 5 ,10 ,3 , 1, 1, 4, 4, 5, 5]
# print(a)
# b = list(set(a))

# print(b)
# counts = []
# for obj in b:
#     count = a.count(obj)
#     counts.append(count)

# print(counts)
















