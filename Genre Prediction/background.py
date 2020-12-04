import cv2
import numpy as np
from matplotlib import pyplot as plt
import argparse
import imutils

def background(input_video):

    probability = [] # Probability is a list which returns the probability of each background. -- 1 for the detected background and 0 for rest backgrounds.
    probabilities = [] # Probabilities is a list which stores the probability of all split videos.


    cap = cv2.VideoCapture(input_video)

    dark = 0 # For storing number of times a frame is dark.
    light = 0 # For storing number of times a frame is light.

    property_id = int(cv2.CAP_PROP_FRAME_COUNT) 
    total_no_of_frames = int(cv2.VideoCapture.get(cap, property_id)) 

    fps = cap.get(cv2.CAP_PROP_FPS)

    seconds_interval = fps * 10
    no_of_loops = int(total_no_of_frames // seconds_interval) + 1

    loops_covered = 0 # Tracks the number of 10 seconds interval covered.

    limit = 0 # A variable used to wait until seconds_interval is reached

    labels = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        limit += 1

        frame = imutils.resize(frame, width = 400)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        blur = cv2.blur(gray, (5, 5)) 

        if cv2.mean(blur)[0] <= 127:
            cv2.putText(frame, "DARK", (frame.shape[1] // 2, frame.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)  
            dark += 1
            # The range for a pixel's value in grayscale is (0-255), dark for less than 127 and light for more than 127

        elif cv2.mean(blur)[0] > 127:
            cv2.putText(frame, "LIGHT", (frame.shape[1] // 2, frame.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            light += 1

        if limit == int(seconds_interval):

            loops_covered += 1

            values_for_each_background = {"Dark" : dark, "Light" : light}

            max_background = max(values_for_each_background, key = values_for_each_background.get)

            if max_background == 'Dark':
                probability = [0.0, 1.0]

            elif max_background == 'Light':
                probability = [1.0, 0.0]

            labels.append(max_background)
            probabilities.append(probability)
            dark = 0
            light = 0
            limit = 0

        # cv2.imshow('frame', frame)

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    cv2.destroyAllWindows()
    if limit < seconds_interval:
        loops_covered += 1

        values_for_each_background = {"Dark" : dark, "Light" : light}

        max_background = max(values_for_each_background, key = values_for_each_background.get)

        if max_background == 'Dark':
            probability = [0.0, 1.0]

        elif max_background == 'Light':
            probability = [1.0, 0.0]

        labels.append(max_background)
        probabilities.append(probability)
        dark = 0
        light = 0

    print("Background Labels:", labels)
    print("Background Probabilities ", probabilities)
    print("Length of background probabilities", len(probabilities))

    return [labels, probabilities]