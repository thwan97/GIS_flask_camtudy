

import cv2
import face_recognition
import numpy as np


camera = cap = cv2.VideoCapture(0,cv2.CAP_DSHOW) # open webcam

def preview_camera():
    while True:
        ## read the camera frame
        success,frame=camera.read()
        if not success:
            break
        else:
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()

        yield(b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')