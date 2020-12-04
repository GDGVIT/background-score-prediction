from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np
import argparse
from collections import Counter 
import time
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

def Emotion_Detection(emotion_model_path, prototext, model, input_video, c):

    net = cv2.dnn.readNetFromCaffe(prototext, model)
    model=Sequential()

    model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(48,48,1)))
    model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7,activation='softmax'))

    model.load_weights(emotion_model_path)
    #print(model.summary())
    emotion_dict={0:"Angry",1:"Disgusted",2:"Fearful",3:"Happy",4:"Neutral",5:"Sad",6:"Surprised"}

    cap = cv2.VideoCapture(input_video)

    property_id = int(cv2.CAP_PROP_FRAME_COUNT) 
    total_no_of_frames = int(cv2.VideoCapture.get(cap, property_id)) 
    # print("Total nunmber of frames in the video are", total_no_of_frames)

    fps = cap.get(cv2.CAP_PROP_FPS)
    # print("Frames per second for this video is : {0}".format(fps))
    seconds_interval = fps * 10
    no_of_loops = int(total_no_of_frames // seconds_interval) + 1
    # print("No of loops to be covered", no_of_loops)
    loops_covered = 0 # Tracks the number of 10 seconds interval covered.

    limit = 0 # A variable used to wait until seconds_interval is reached
    count = 0
    
    labels = []
    probabilities = []
    sum_of_emotions_probabilities = np.array([0, 0, 0, 0, 0, 0, 0])
    cv2.ocl.setUseOpenCL(False)
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = imutils.resize(frame, width = 400)
        limit += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        (h,w) = frame.shape[:2]
        
        blob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)),1.0,(300,300),(104.0,177.0,123.0))

        net.setInput(blob)
        detections = net.forward()
        # print(detections)
        no_of_person = 0

        for i in range(0,detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence < c:
                break

            # print(confidence)
            no_of_person += 1
            box = detections[0, 0, i, 3 : 7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            count += 1

            text = "{:.2f}%".format(confidence*100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0,0,255), 1)
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        # coordinates_for_multi_person.append([startX, startY, endX, endY])
        # print(coordinates_for_multi_person)
        
        # Extract the ROI of the face from the grayscale image
            roi = gray[startY : endY, startX : endX]
            # print(roi.shape)
            if roi.shape[0] == 0 or roi.shape[1] == 0:
                break
            roi = np.expand_dims(np.expand_dims(cv2.resize(roi, (48, 48)), -1), 0)

            preds = model.predict(roi)[0]
            pred1 = np.array(preds)
            # print(pred1)

            sum_of_emotions_probabilities = sum_of_emotions_probabilities + pred1
            # print("Limit", limit)
            # if limit == int(seconds_interval):
            #     loops_covered += 1
            #     sum_of_emotions_probabilities = sum_of_emotions_probabilities / seconds_interval
            #     sum_of_emotions_probabilities = [ '%.2f' % elem for elem in sum_of_emotions_probabilities]
            #     sum_of_emotions_probabilities = [float(elem) for elem in sum_of_emotions_probabilities]
            #     emotion_probability = np.max(sum_of_emotions_probabilities)
            #     label = EMOTIONS[emotion_probability.argmax()]
            #     labels.append(label)
            #     probabilities.append(sum_of_emotions_probabilities)
            #     print(sum_of_emotions_probabilities)
            #     print(label)
            #     sum_of_emotions_probabilities = np.array([0, 0, 0, 0, 0, 0, 0])
            #     limit = 0
        # print(sum_of_emotions_probabilities)
        if limit == int(seconds_interval):
            loops_covered += 1
            sum_of_emotions_probabilities = list(sum_of_emotions_probabilities)
            if sum_of_emotions_probabilities == [0, 0, 0, 0, 0, 0, 0]:
                probabilities.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                labels.append("No Person Detected")
                # print("10 seconds paased")
            else:
                sum_of_emotions_probabilities = np.array(sum_of_emotions_probabilities)
                sum_of_emotions_probabilities = sum_of_emotions_probabilities / count
                sum_of_emotions_probabilities = [ '%.2f' % elem for elem in sum_of_emotions_probabilities]
                sum_of_emotions_probabilities = [float(elem) for elem in sum_of_emotions_probabilities]
                maxindex=int(np.argmax(sum_of_emotions_probabilities))
                label=emotion_dict[maxindex]
                # print(label)
                labels.append(label)
                probabilities.append(sum_of_emotions_probabilities)
                # print(sum_of_emotions_probabilities)
                # print(label)
                sum_of_emotions_probabilities = np.array([0, 0, 0, 0, 0, 0, 0])
                count = 0
                # print("10 seconds passed")
            limit = 0




        # print("Confidence score is", confidence)
        # print("No of person", no_of_person)
        # print("limit", limit)
        # if no_of_person == 0 and limit == int(seconds_interval):
        #     loops_covered += 1
        #     probabilities.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        #     labels.append("No Person Detected")
        #     limit = 0
            # print("Count is", count)
            # print("No person detected")

            # print("One person detection over ----", label)
        # print("Number of people in the frame", no_of_person)
        # print("Frame number", total_no_of_frames) 

        # cv2.imshow('your_face', frame)
        # if cv2.waitKey(1)  & 0xFF == ord('q'):
        #     break

    cap.release()
    cv2.destroyAllWindows()
    if limit < seconds_interval:
        loops_covered += 1
        # print(sum_of_emotions_probabilities)
        # print("Sum_of_Emotional_Probabilities", sum_of_emotions_probabilities)
        sum_of_emotions_probabilities = list(sum_of_emotions_probabilities)
        if sum_of_emotions_probabilities == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]:
            labels.append("No person detected")
            probabilities.append([0.0] * 7)
        else:
            sum_of_emotions_probabilities = np.array(sum_of_emotions_probabilities)
            sum_of_emotions_probabilities = sum_of_emotions_probabilities / count
            sum_of_emotions_probabilities = [ '%.2f' % elem for elem in sum_of_emotions_probabilities]
            sum_of_emotions_probabilities = [float(elem) for elem in sum_of_emotions_probabilities]
            maxindex = int(np.argmax(sum_of_emotions_probabilities))
            label = emotion_dict[maxindex]
            labels.append(label)
            probabilities.append(list(sum_of_emotions_probabilities))
    # print("Total nunmber of frames in the video are", total_no_of_frames)
    # print("Frames per second for this video is : {0}".format(fps))
    # print("No of loops to be covered", no_of_loops)
    # print("Loops covered", loops_covered)
    # print("Emotion Labels", labels)
    # print("Emotion Probabilities ", probabilities)
    # print("Length of Emotion Probabilities", len(probabilities))

    return [labels, probabilities]