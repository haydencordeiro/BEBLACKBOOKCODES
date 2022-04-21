
from asyncio import constants
from glob import glob
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import time
import socketio
from multiprocessing import Process, Value
from flask import Flask 
from flask_socketio import SocketIO, send
from pathlib import Path
import cv2
import numpy
from fer import FER
import PIL
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from io import BytesIO 
from tensorflow.keras.models import load_model

model = load_model('CNNmodelGray')

sio = socketio.Client()

sio.connect('http://127.0.0.1:5000')


def calc_emotion(img):
    try:
        x = cv2.resize(img, (150, 150))
        x= cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        x=x.reshape(1,150,150,1)
        return(model.predict(x)[0][0])

    except:
        return "Ignore"

while True:

        it= os.listdir(r"D:\BEFINAL\datastream\deep")
        if it:
            it.sort(key=lambda x:os.path.getctime("D:/BEFINAL/datastream/deep"+"/"+ x))
        else:
            it=[]
        for entry in it:
            try:
                temp=Path(entry).name.split("-")
                person_id=temp[0]
                uid=temp[1][:-4]
                with open("D:/BEFINAL/datastream/deep"+"/"+entry, 'rb') as f:
                    check_chars = f.read()[-2:]
                if check_chars != b'\xff\xd9':
                    print('Not complete image deep')
                    continue
                else:
                    res =calc_emotion("D:/BEFINAL/datastream/deep/"+entry)
                    buff =  BytesIO()
                    buff.write(f.read())
                    buff.seek(0)
                    temp_img = numpy.array(PIL.Image.open(buff), dtype=numpy.uint8)
                    img = cv2.cvtColor(temp_img, cv2.COLOR_RGB2BGR)
                    res =calc_emotion(img)
                if res=="Ignore":
                    pass
                else:
                    pass
                    
                    sio.send(f"deep-{person_id}-{res}-{uid}")
                os.remove("D:/BEFINAL/datastream/deep"+"/"+entry)
            except Exception as e:
                print(e,"deep")
