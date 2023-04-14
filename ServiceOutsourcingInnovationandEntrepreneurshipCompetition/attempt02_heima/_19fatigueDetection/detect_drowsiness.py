# @Version: python3.10
# @Time: 2023/4/14 15:38
# @Author: MaxBrooks
# @Email: 15905898514@163.com
# @File: detect_drowsiness.py
# @Software: PyCharm
# @User: chent

import argparse
import time
from threading import Thread

import cv2
import dlib
import imutils
import playsound
from imutils import face_utils
from imutils.video import VideoStream
from scipy.spatial import distance as dist

EYE_AR_THRESH = 0.3
EYE_AR_CONSE_FRAMES = 48

COUNTER = 0
ALARM_ON = False

ap = argparse.ArgumentParser()
ap.add_argument('-p', '--shape-predictor', required=True,
    help="path to facial landmark predictor")
ap.add_argument('-a', '--alarm', type=str, default="",
    help="path alarm .WAV file")
ap.add_argument('-w', '--webcam', type=int, default=0,
    help="index of webcam on system")
args = vars(ap.parse_args())

def sound_alarm(path):
    playsound.playsound(path)


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear


print("[INFO] loading facial landmarks predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args['shape_predictor'])

(lStart, IEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
print('[INFO] starting video stream thread...')
vs = VideoStream(src=args['webcam']).start()
time.sleep(1.0)

while True:

    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:IEnd]
        rightEye = shape[rStart:rEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < EYE_AR_THRESH:
            COUNTER += 1

            if COUNTER >= EYE_AR_CONSE_FRAMES:
                if not ALARM_ON:
                    ALARM_ON = True


                    if args['alarm'] != '':
                        t = Thread(target=sound_alarm,
                                   args=(args['alarm'],))
                        t.deamon = True
                        t.start()

                cv2.putText(frame, 'DROWSINESS ALARM!', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            COUNTER = 0
            ALARM_ON = False


        cv2.putText(frame, 'ERA: {:.2f}'.format(ear), (300, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitkey(1) & 0xFF

    if key == ord('q'):
        break

cv2.destroyAllWindows()
vs.stop()



