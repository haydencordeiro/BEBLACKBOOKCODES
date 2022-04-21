
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from tensorflow import keras
import os
import time
import socketio
from multiprocessing import Process, Value
from flask import Flask 
from flask_socketio import SocketIO, send
from pathlib import Path




sio = socketio.Client()
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

sio.connect('http://127.0.0.1:5000')
model_path = "emotion.h5"
loaded_model = keras.models.load_model(model_path)
class_names = ['Angry','Disgusted','Scared','Happy','Sad','Surprised']
d={"1":[],"2":[]}
while True:

        it= os.listdir(r"D:\BEFINAL\datastream\emo")
        if it:
            it.sort(key=lambda x:os.path.getctime("D:/BEFINAL/datastream/emo"+"/"+ x))
        else:
            it=[]
        for entry in it:
            try:
                temp=Path(entry).name.split("-")
                person_id=temp[0]
                uid=temp[1][:-4]
                with open("D:/BEFINAL/datastream/emo"+"/"+entry, 'rb') as f:
                    check_chars = f.read()[-2:]
                if check_chars != b'\xff\xd9':
                    print('Not complete image')
                    continue
                else:
                    image = cv2.imread("D:/BEFINAL/datastream/emo"+"/"+entry, 1)


                image_fromarray = Image.fromarray(image, 'RGB')
                resize_image = image_fromarray.resize((128, 128))
                expand_input = np.expand_dims(resize_image,axis=0)
                input_data = np.array(expand_input)
                input_data = input_data/255

                pred = loaded_model.predict(input_data)
                result = pred.argmax()
                # print(class_names[result])
                pred=pred[0]
                temp_pred=[]
                mi=np.min(pred)
                ma=np.max(pred)
                for i,j in enumerate(pred):
                    temp=((j-mi)/(ma-mi))
                    if i in [3,5]:
                        temp*=-1
                    temp_pred.append(round(temp,6))
                sio.send(f"emo-{person_id}-{ round((sum(temp_pred)+2)/6,6)}-{uid}")
                os.remove("D:/BEFINAL/datastream/emo"+"/"+entry)
            except Exception as e:
                print(e)

