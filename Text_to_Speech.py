from gtts import gTTS
import os

#fh=open("Read1.txt","r")
fh=open("Read2.txt","r")
#fh=open("Read3.txt","r")
myText=fh.read().replace("\n"," ")

language='en'

output=gTTS(text=myText , lang=language, slow =False)

#output.save("output.mp3")
fh.close()
os.system("start output.mp3")


