from background import background
from Emotion_Detection2 import Emotion_Detection
import cv2
import numpy as np
import pandas as pd
import pickle

emotion_model_path = 'Emotion Detection_Files/model.h5' # Path for Emotion Classification Model
prototext = 'Emotion Detection_Files/deploy.prototxt.txt' # Prototxt file for face detection
model = 'Emotion Detection_Files/res10_300x300_ssd_iter_140000.caffemodel' # Model for face Detection
input_video = 'Example_Videos/fighting.mp4' # Path for video
c = 0.7 # Confidence score for detecting the face of a person

background_labels, background_probabilities = background(input_video)

emotion_labels, emotion_probabilities = Emotion_Detection(emotion_model_path, prototext, model, input_video, c)

rows = []

if len(emotion_probabilities) == len(background_probabilities):
    for i in range(0, len(emotion_probabilities)):
        rows.append(emotion_probabilities[i] + background_probabilities[i])  # Concatenating the two lists.

print(rows)
print("Done")

model_path = 'finalized_model.sav'
loaded_model = pickle.load(open(model_path, 'rb'))
if rows != []:
    df = pd.DataFrame(rows)

    predictions = list(loaded_model.predict(df.values))

    Genres = {0 : 'Horror', 1 : 'Action' , 2 : 'Comedy', 3 : 'Romantic'}

    predictions = list(map(Genres.get, predictions))
    print(predictions)

else:
    print("Please check")