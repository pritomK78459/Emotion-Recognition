# import packages

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, BatchNormalization, MaxPooling2D, Dropout, Input
import queue
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='script to run the emotion recorgnition model on the given video')
    parser.add_argument('--video', help='path to the video on which the model has to be run.', type=str)
    parser.add_argument('--image', help='path to the image on which the model has to be run', type=str)
    parser.add_argument('--camera', help='run the model on feed given by the device camera', action='store_true')
    parser.set_defaults(video=None, image=None,camera=False)
    args = parser.parse_args()

    model = tf.keras.models.load_model('models/1')  # load the model weights  
    

    # initialize the face detection system 
    face_cascade = cv2.CascadeClassifier(
        "face_detection_data/haarcascades/haarcascade_frontalface_default.xml"
    )

    def capture_from_video(video_medium):

        # this function will run the emotion detection model on the video provided

        average_emotion = []    # predicted emotion for 25 frames
        emotion_label = 'none'  # initialize emotion label

        video = cv2.VideoCapture(video_medium)   # the video on which the model will be ran

        frame_width = int(video.get(3))
        frame_height = int(video.get(4))
        
        size = (frame_width, frame_height)

        result = cv2.VideoWriter('result_video.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)

        while cv2.waitKey(1) == -1:     # wait until the keypress
            success, frame = video.read()   # frame and success status

            if success:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert the captured frame to gray

                # get all the faces in the current frame with minimum size of 120x120 pixels
                faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(120, 120))

                for (x, y, w, h) in faces:  # iterate through all the faces
                    image = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)    # draw a rectangle around the face
                    # put the current emotion of that face above the box as a label
                    cv2.putText(image, f'Emotion: {emotion_label} ', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                
                    roi_gray = gray[y : y + h, x : x + h]   # slice out the region of interest 
                    roi_gray = cv2.resize(roi_gray, (48,48))    # resize the image according to the size required be the model
                    image = roi_gray
                    image = np.expand_dims(image, 0)    # add one color channel as it is a grayscale image
                    image = tf.convert_to_tensor(image, dtype=tf.float64)   # convert to tensor

                    predicted_emotions = model.predict(image)   # pass the image to the model
                    emotion = np.argmax(predicted_emotions[0])  # get the prediction list and get the index of the max 
                    
                    if len(average_emotion) > 25:   # check if average emotion count has gone above 25
                        average_emotion.pop()       # if more than 25 remove the last label
                    else:
                        average_emotion.append(emotion) # if less then add the most recent predicted emotion

                    emotion = int(np.mean(average_emotion)) # take the average of the emotion

                    label_map = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral') # list of emotions
                    emotion_label = label_map[emotion]  # convert the integer label to human understandable string
                    

                    print(emotion_label)    
                
                result.write(frame)
                cv2.imshow("Face Detection", frame) # display the frame
                 
                cv2.waitKey(10) # delay the next iteration so that the video runs at a proper speed

    def use_image():

        # this function will run the emotoin detection model on the provided image
        
        emotion_label = 'None'
        given_image = cv2.imread(args.image)
        given_image = cv2.resize(given_image, (770, 770))
        gray = cv2.cvtColor(given_image, cv2.COLOR_BGR2GRAY)  # convert the captured frame to gray
        

        # get all the faces in the current frame with minimum size of 120x120 pixels
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(120, 120))

        for (x, y, w, h) in faces:  # iterate through all the faces
            image = cv2.rectangle(given_image, (x , y ), (x + w , y + h ), (255, 0, 0), 2)    # draw a rectangle around the face
            # put the current emotion of that face above the box as a label
            
            roi_gray = gray[y : y + h , x : x + h ]   # slice out the region of interest 
            roi_gray = cv2.resize(roi_gray, (48,48))    # resize the image according to the size required be the model
            image = roi_gray
            image = np.expand_dims(image, 0)    # add one color channel as it is a grayscale image
            image = tf.convert_to_tensor(image, dtype=tf.float64)   # convert to tensor
            image_batch = np.expand_dims(image, 0)
            print(image_batch.shape)
            print(image.shape)

            predicted_emotions = model.predict(image)   # pass the image to the model
            emotion = np.argmax(predicted_emotions[0])  # get the prediction list and get the index of the max 
            

            label_map = ('angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise') # list of emotions
            emotion_label = label_map[emotion]  # convert the integer label to human understandable string
            
            cv2.putText(given_image, f'Emotion: {emotion_label} ', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            
            print(emotion_label)    
            
        file_name = f"sample_images/{emotion_label}.jpg"
        cv2.imwrite(file_name, given_image)
        cv2.imshow("Emotion Detection", given_image) # display the frame 
        cv2.waitKey(0) # delay the next iteration so that the video runs at a proper speed

    if args.video is not None:
        capture_from_video(args.video)  # use the given video
    
    elif args.image is not None:
        use_image()                     # use the given image

    elif args.camera:
        capture_from_video(0)           # capture from device camera