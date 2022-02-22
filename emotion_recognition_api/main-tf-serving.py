import re
from typing import Dict
from unittest import result
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, BatchNormalization, MaxPooling2D, Dropout, Input
import cv2
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import requests


model = tf.keras.models.load_model('models/1')

# initialize the face detection system 
face_cascade = cv2.CascadeClassifier(
        "face_detection_data/haarcascades/haarcascade_frontalface_default.xml"
)

app = FastAPI()

origins = [
    "http://localhost:3000",
    "localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

endpoint = "http://localhost:8501/v1/models/emotion_recognition/versions/1:predict"


def read_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


def predict_using_image(image: np.ndarray):
    # this function will run the emotoin detection model on the provided image

    emotion_label = 'None'
    given_image = image
    given_image = cv2.resize(given_image, (770, 770))
    gray = cv2.cvtColor(given_image, cv2.COLOR_BGR2GRAY)  # convert the captured frame to gray

    # get all the faces in the current frame with minimum size of 120x120 pixels
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(120, 120))

    for (x, y, w, h) in faces:  # iterate through all the faces
        image = cv2.rectangle(given_image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # draw a rectangle around the face
        # put the current emotion of that face above the box as a label

        roi_gray = gray[y: y + h, x: x + h]  # slice out the region of interest
        roi_gray = cv2.resize(roi_gray, (48, 48))  # resize the image according to the size required be the model
        image = roi_gray
        image = np.expand_dims(image, 0)  # add one color channel as it is a grayscale image
        image = tf.convert_to_tensor(image, dtype=tf.float64)  # convert to tensor
        image = image.numpy()
        image_batch = np.expand_dims(image, -1)

        json_data = {
            "instances": image_batch.tolist()
        }
        response = requests.post(endpoint, json=json_data)


        
        # predicted_emotions = model.predict(image)  # pass the image to the model
        predictions = response.json()["predictions"]

        emotion = np.argmax(predictions[0])  # get the prediction list and get the index of the max
        
        label_map = ('angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise')  # list of emotions
        emotion_label = label_map[emotion]  # convert the integer label to human understandable string
        
        result_image = cv2.putText(given_image, f'Emotion: {emotion_label} ', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (36, 255, 12), 2)
        
        return emotion_label, predictions[0]

@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict:
    
    image = read_image(await file.read())   # read the image

    emotion_label, predictions = predict_using_image(image) # predict using the model
    confidence = np.max(predictions)    # get the confidence score

    return dict({
        "class": emotion_label,
        "confidence": confidence
    })


if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000)   # run the app through uvicorn