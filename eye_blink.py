

from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2
from scipy.spatial import distance as dist


def sound_alarm(path):
    # play an alarm sound
    playsound.playsound(path)


def eye_aspect_ratio(eye):
    
    
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    
    ear = (A + B) / (2.0 * C)
    
    return ear


ap = argparse.ArgumentParser()
ap.add_argument("-p",
                "--shape-predictor",
                required=True,
                help="path to facial landmark predictor")
ap.add_argument("-a",
                "--alarm",
                type=str,
                default="",
                help="path alarm .WAV file")

args = vars(ap.parse_args())

EYE_ASPECT_RATIO_THRESH = 0.3
EYE_ASPECT_RATIO_CONSEC_FRAMES = 48

COUNTER = 0
ALARM_ON = False

print("loading facial landmark predictor")
detector = dlib.get_frontal_face_detector()



predictor = dlib.shape_predictor(args["shape_predictor"])


(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]




cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
score = 0
thicc = 2


while True:
    
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    rects = detector(gray, 0)

    for rect in rects:
       
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftAspectRatio = eye_aspect_ratio(leftEye)
        rightAspectRatio = eye_aspect_ratio(rightEye)
        aspect_ratio = (leftAspectRatio + rightAspectRatio) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        
        if aspect_ratio < EYE_ASPECT_RATIO_THRESH:
            COUNTER += 1
           
            if COUNTER >= EYE_ASPECT_RATIO_CONSEC_FRAMES:
                
                if not ALARM_ON:
                    ALARM_ON = True
                   
                    if args["alarm"] != "":
                        t = Thread(target=sound_alarm, args=(args["alarm"], ))
                        t.deamon = True
                        t.start()
                
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            COUNTER = 0
            ALARM_ON = False

        cv2.putText(frame, "Eye Aspect Ratio: {:.2f}".format(aspect_ratio), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    
    if key == 27:
        break

cv2.destroyAllWindows()
