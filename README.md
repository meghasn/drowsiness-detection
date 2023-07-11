# Drowsiness Detection for Road Safety

This code implements a drowsiness detection system using computer vision techniques. The purpose of this system is to detect drowsiness among drivers, contributing to road safety. It utilizes facial landmarks and eye aspect ratios to determine if the driver is displaying signs of drowsiness.

## Requirements
To run this code, you need the following dependencies:

Python 3.x
OpenCV
imutils
playsound
dlib
scipy
Setup
Install the required dependencies using the following command:

Copy code
pip install opencv-python imutils playsound dlib scipy


If you want to use a custom alarm sound, provide the path to the .WAV file using the -a or --alarm argument when running the script.


## To run the drowsiness detection system, execute the following command:

python drowsiness_detection.py -p shape_predictor_68_face_landmarks.dat

## Optional arguments:

-p or --shape-predictor: Path to the facial landmark predictor file (required).
-a or --alarm: Path to the custom alarm sound file (optional).

## Description
The code initializes the required modules and sets the necessary constants for drowsiness detection. It captures video frames from the webcam and processes them in a loop. The facial landmarks are detected using the shape predictor, and eye aspect ratios are calculated based on the positions of the eyes. If the average eye aspect ratio falls below a threshold for a certain number of consecutive frames, it triggers a drowsiness alert.

The system draws contours around the eyes on the video frame and displays the eye aspect ratio. When drowsiness is detected, a warning message is shown on the frame, and an alarm sound can be played.

The code terminates when the user presses the Esc key.

## Use Case
This algorithm was developed to implement an interactive project aimed at educating students about the importance of computer vision in road safety. By detecting drowsiness among drivers in real-time, it serves as a tool to raise awareness and prevent accidents caused by driver fatigue.

Please note that this code is a simplified implementation and may require further optimization or customization to suit specific needs or production environments.

For more information and inquiries, please contact Megha Sudhakaran Nair (email: megha99.sudhakaran@gmail.com).
