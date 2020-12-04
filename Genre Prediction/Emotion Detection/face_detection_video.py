from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2

ap = argparse.ArgumentParser()

ap.add_argument("-p", "--prototext", required=True, help="path to Caffe 'deploy' prototext file")
ap.add_argument("-m", "--model", required=True, help="Path to Caffe pre-trained model")
ap.add_argument("-i", "--input", required = True, help = "Path to Input Video")
ap.add_argument("-c","--confidence",type=float,default=0.5,help="minimum probability to filter weak detections")

args=vars(ap.parse_args())

net = cv2.dnn.readNetFromCaffe(args["prototext"], args["model"])

if args["input"] == "self":
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(str(args['input']))

time.sleep(2.0)

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame,width=400)

    if not ret:
        break

    (h,w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)),1.0,(300,300),(104.0,177.0,123.0))

    net.setInput(blob)
    detections=net.forward()

    for i in range(0,detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence < args['confidence']:
            continue
        box = detections[0,0,i,3:7]*np.array([w,h,w,h])
        (startX,startY,endX,endY) = box.astype("int")

        text = "{:.2f}%".format(confidence*100)
        y = startY-10 if startY-10>10 else startY+10
        cv2.rectangle(frame,(startX,startY),(endX,endY),(0,0,255),1)
        cv2.putText(frame,text,(startX,y),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,0),1)
    cv2.imshow("Frame",frame)
    key = cv2.waitKey(1)&0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
