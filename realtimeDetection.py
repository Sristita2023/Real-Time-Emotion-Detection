import cv2
from keras.models import model_from_json
import numpy as np
import random
json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

model.load_weights("emotiondetector.h5")
haar_file=cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade=cv2.CascadeClassifier(haar_file)

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1,48,48,1)
    return feature/255.0

webcam=cv2.VideoCapture(0)

labels = {0 : "Angry", 1 : "Disgust", 2 : "Fear", 3 : "Happy", 4 : "Neutral", 5 : "Sad", 6 : "Surprised"}
while True:
    i,im=webcam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(im,1.3,5)
    try: 
        for (p,q,r,s) in faces:
            image = gray[q:q+s,p:p+r]
            cv2.rectangle(im,(p,q),(p+r,q+s),(255,0,0),2)
            image = cv2.resize(image,(48,48))
            img = extract_features(image)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]
            
            cv2.putText(img = im, text = '% s' %(prediction_label), org = (p-10, q-10),fontFace = cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale = 1, color = (125, 246, 55), thickness=1)
        # cv2.imshow("Output",im)
        cv2.imshow('Output', cv2.resize(im, (1500,960), interpolation=cv2.INTER_CUBIC))
        # cv2.waitKey(27)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except cv2.error:
        pass
