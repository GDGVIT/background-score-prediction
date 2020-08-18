import numpy as np
import argparse
import imutils
import sys
import cv2
print("All libraries loaded")
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required = True, help = "path to trained human activitt recognition model")
ap.add_argument("-c", "--classes", required = True, help = "Path to class labels file")
ap.add_argument("-i", "--input", required = True, help = "Optional path to video file")

args = vars(ap.parse_args())
CLASSES = open(args["classes"]).read().strip().split('\n')
SAMPLE_DURATION = 16
SAMPLE_SIZE = 112

print("Loading Human activity recognition model")
net = cv2.dnn.readNet(args["model"])
vs = cv2.VideoCapture(args["input"] if args["input"] else 0)

number_of_times_while_executed = 0

while True:
    # initialize the batch of frames that will be passed through the
	# model
    frames = []

    for i in range(0, SAMPLE_DURATION):
        (grabbed, frame) = vs.read()

        if not grabbed:
            print("No frmae read from stream - exiting")
            sys.exit(0)
        
        frame = imutils.resize(frame, width = 400)
        frames.append(frame)
    number_of_times_while_executed += 1
    # print("Number of times the while loop is executed:", number_of_times_while_executed)
    
    blob = cv2.dnn.blobFromImages(frames,
    1.0, (SAMPLE_SIZE, SAMPLE_SIZE), (114.7748, 107.7354, 99.4750), swapRB = True, crop = True)
    blob = np.transpose(blob, (1, 0, 2, 3))
    blob = np.expand_dims(blob, axis = 0)

    #(1, 3, 16, 112, 112)
    # 1 is the batch dimension a single data point is passed through a network, a “data point” in this 
    # context means the N frames that will be passed through the network to 
    # obtain a single classification).
    # 16 is the number of Blobs
    net.setInput(blob)
    outputs = net.forward()
    label = CLASSES[np.argmax(outputs)]

    for frame in frames:
        # draw the predicted activity on the frame
        cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
        cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Activity Recognition", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break








