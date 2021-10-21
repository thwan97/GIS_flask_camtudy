

from flask import Flask, render_template, Response
import cv2
import face_recognition
import numpy as np


# -- 객체 생성 부분 --
app=Flask(__name__)


camera = cap = cv2.VideoCapture(0) # open webcam

# Load a sample picture and learn how to recognize it.
krish_image = face_recognition.load_image_file("static/images/krish.jpg")
krish_face_encoding = face_recognition.face_encodings(krish_image)[0]

# Load a second sample picture and learn how to recognize it.
bradley_image = face_recognition.load_image_file("static/images/bradley.jpg")
bradley_face_encoding = face_recognition.face_encodings(bradley_image)[0]
# -- 객체 생성 부분 끝 --


# Create arrays of known face encodings and their names


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



 # 메인페이지로 라우팅 (페이지) 

@app.route('/')
def main_page_route():
    return render_template('main_page.html')

# 메인에서 미리보기 카메라 켜지는 부분 (기능)
@app.route('/preview_camera')
def preview_camera_route():
    return Response(preview_camera(), mimetype='multipart/x-mixed-replace; boundary=frame')





import cv2 
import dlib 
from scipy.spatial import distance
from imutils import face_utils
import mediapipe as mp
import math

#hand
import views.HandTrackingModule as htm

def htm_camera():

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('face_finger-master/shape_predictor_68_face_landmarks.dat')

    #hand
    hand_detector = htm.handDetector(detectionCon=0.8)
    tipIds = [4, 8, 12, 16, 20] # 엄지 검지 중지 약지 소지

    def eye_aspect_ratio(eye): ##눈의 크기를 return 해주는 함수
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])

        C = distance.euclidean(eye[0], eye[3])
        eye = (A + B) / (2.0 * C)

        return eye

    def face_size(val1 , val2, face_landmarks):
        x = abs(face_landmarks.part(val1).x - face_landmarks.part(val2).x)
        y = abs(face_landmarks.part(val1).y - face_landmarks.part(val2).y)
        return (x ** 2 + y ** 2) ** (1/2)

    #얼굴포인트 -  손가락포인트거리
    def dist_face_finger(lmList, face_landmarks, val1, val2):
        x = abs(face_landmarks.part(val1).x - lmList[val2][1])
        y = abs(face_landmarks.part(val1).y - lmList[val2][2])
        return (x ** 2 + y ** 2) ** (1/2)

    def angle(a, b):
        x = a.x - b.x
        y = a.y - b.y
        radian = math.atan2(y, x)
        degree = radian * 180 / 3
        return degree

    count = 0
    count2 = 0
    count3 = 0
    count4 = 0
    total = 0
    first = 0
    height = -1
    width = -1
    face_landmarks = 0

    while True:
        success,img = cap.read()

        if not success:
            break

        else:
            imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            faces = detector(imgGray)
            results = pose.process(img)
            try:
                landmarks = results.pose_landmarks.landmark
                right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value]
                right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
                left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value]
                left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]

                if right_wrist.y < right_eye.y and left_wrist.y < left_eye.y and abs(right_elbow.x - right_shoulder.x) < 0.1 and abs(right_wrist.x - right_elbow.x) < 0.1 and abs(left_elbow.x - left_shoulder.x) < 0.1 and abs(left_wrist.x - left_elbow.x) < 0.1 :
                    print('만세!!')
     
            except:
                pass
            

            for face in faces:
                face_landmarks = predictor(imgGray,face)

                landmarks = face_utils.shape_to_np(face_landmarks) ## 얼굴 landmark 좌표값을 가지고 있는 변수
                leftEye = landmarks[42:48]
                rightEye = landmarks[36:42]

                leftEye = eye_aspect_ratio(leftEye)
                rightEye = eye_aspect_ratio(rightEye)

                eye = (leftEye + rightEye) / 2.0
                

                if eye<0.3:
                    count+=1
                else:
                    if count>=3:
                        total+=1

                    count=0
                if first == 0:
                    fis_width = face_size(1, 16, face_landmarks) ## 첫 화면에서 얼굴의 가로 크기를 저장
                else:
                    width = face_size(1, 16, face_landmarks) ## 화면 속의 얼굴 크기를 가짐
                n = 27
                m = 8

                if first == 0:
                    fis_height = face_size(27, 8, face_landmarks) ##첫 화면 세로 크기
                    fis_eyebrow_y = face_landmarks.part(21).y
                    first += 1
                else:
                    height = face_size(27, 8, face_landmarks) ## 화면속 세로 크기
                    eyebrow_y = face_landmarks.part(21).y
                    first += 0.1
            
            #hand
            img = hand_detector.findHands(img)
            lmList = hand_detector.findPosition(img, draw=False)
            degree2 = hand_detector.angle(img, draw=False)

            
            if len(lmList) != 0 and face_landmarks != 0:
                # 새끼손가락 얼굴안 & 손 키포인트각들의 합
                if lmList[20][1] > face_landmarks.part(1).x and lmList[20][1] < face_landmarks.part(15).x and degree2 > 100 and degree2 < 300:
                    count2 += 1
                    if count2 == 50:
                        print("손 얼굴안")
                        count2 = 0
                else:
                    count2 = 0        
                
                dist_1 = dist_face_finger(lmList, face_landmarks, 40, 8) #왼쪽눈 검지 거리
                dist_2 = dist_face_finger(lmList, face_landmarks, 47, 8) #오른쪽눈 검지 거리
                if dist_1 < 15 or dist_2 < 15:
                    count3 += 1
                    if count3 == 30:
                        print("눈비빔")
                        count3 = 0
                else:
                    count3 = 0

                # 입에 손가락    
                dist_3 = []
                for num in [8,12,16,20]:
                    dist_3.append(dist_face_finger(lmList, face_landmarks, 66, num))

                if ((dist_3[0] < 20 or dist_3[1] < 20 or dist_3[2] < 20 or dist_3[3] < 20) and degree2 > 400 and degree2 < 900):
                    count4 += 1
                    if count4 == 40:
                        print("입에 손가락")
                        count4 = 0
                else:
                    count4 = 0
                # print(dist_3[0])


                
            if first > 1:
                if (height - fis_height + width - fis_width) > 50: ## 첫 화면 얼굴 크기보다 나중 화면 얼굴 크기가 일정 값 이상 커질때
                    cv2.putText(img, "go back!", (100, 100),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)    
                if(eyebrow_y - fis_eyebrow_y) > 50: ## 첫 화면 눈썹 y좌표보다 나중 화면 눈썹y좌표가 일정 값 이상 커질 때
                    cv2.putText(img, "stretch your back!", (100, 100),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)  
            cv2.putText(img, "Blink Count: {}".format(total), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # if (height > 0) and (width > 0):
            #     print(fis_height, fis_width, height, width)
            # cv2.imshow('Video',img)
            if cv2.waitKey(1) & 0xff==ord('q'):
                break

            ret, buffer = cv2.imencode('.jpg', img)
            img = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')



# 메인 페이지에서 스터디 들어가기 버튼 클릭 -> htm_page로 감 (여기서 htm_camera를 킴) (페이지)
@app.route('/htm_page')
def htm_page_route():
    return render_template('htm_page.html')
@app.route('/htm_camera')
def htm_camera_route():
    return Response(htm_camera(), mimetype='multipart/x-mixed-replace; boundary=img')


if __name__=='__main__':
    app.run(debug=True)