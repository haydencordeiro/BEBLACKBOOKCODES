
import numpy as np
import cv2
from tensorflow.keras.models import load_model


model = load_model('CNNmodelGray')
font = cv2.FONT_HERSHEY_SIMPLEX
cap = cv2.VideoCapture('videos/grejoNotIntrested.mp4')
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        img = frame
        x = cv2.resize(img, (150, 150))
        x= cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        x=x.reshape(1,150,150,1)


        prediciton=(np.argmax(model.predict(x), axis=-1))[0]
        cv2.putText(img,'Intrested' if prediciton==1 else "Not Intrested",(30,30), font, 1,(0,0,0),2)
        cv2.imshow('frame', img)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows