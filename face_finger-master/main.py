import cv2
import time
import os
import dlib
import HandTrackingModule as htm
import numpy as np
import math


num = 0
wCam, hCam = 640, 480
detector_face = dlib.get_frontal_face_detector()
 
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# range는 끝값이 포함안됨   
ALL = list(range(0, 68)) 
# RIGHT_EYEBROW = list(range(17, 22))  
# LEFT_EYEBROW = list(range(22, 27))  
# RIGHT_EYE = list(range(36, 42))  
# LEFT_EYE = list(range(42, 48))  
# NOSE = list(range(27, 36))  
# MOUTH_OUTLINE = list(range(48, 61))  
# MOUTH_INNER = list(range(61, 68)) 
# JAWLINE = list(range(0, 17)) 

index = ALL

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = "FingerImage"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    # print(f'{folderPath}/{imPath}')
    overlayList.append(image)

print(len(overlayList))
pTime = 0

detector = htm.handDetector(detectionCon=0.8)

tipIds = [4, 8, 12, 16, 20] # 엄지 검지 중지 약지 소지

while True:
    success, img = cap.read()
    success2, img2 = cap.read()

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detector_face(img_gray, 1)

    for face in dets:

        shape = predictor(img2, face) #얼굴에서 68개 점 찾기
        kkcy = 0
        lex, rex = 0, 0
        list_points = []
        for p in shape.parts():
            list_points.append([p.x, p.y])

        kkcy = list_points[33][1] # 코끝 y
        lex = list_points[2][0]  # 왼귀 x
        rex = list_points[14][0] # 오른귀 x
        leyex = list_points[41][0]
        leyey = list_points[41][1]
        reyex = list_points[46][0]
        reyey = list_points[46][1]

        radian = math.atan2(leyey - reyey, leyex - reyex)
        degree = radian * 180 / math.pi
        cv2.putText(img, "eye degree: {}".format(abs(degree)), (10, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        list_points = np.array(list_points)  

        # 코끝, 왼쪽볼끝 오른쪽볼끝 턱끝 점 찍기
        pt_pos = (list_points[33][0], list_points[33][1])
        pt_pos2 = (list_points[2][0], list_points[2][1])
        pt_pos3 = (list_points[14][0], list_points[14][1])
        pt_pos4 = (list_points[8][0], list_points[8][1])
        cv2.circle(img, pt_pos, 2, (0, 255, 0), -1)
        cv2.circle(img, pt_pos2, 2, (0, 255, 0), -1)
        cv2.circle(img, pt_pos3, 2, (0, 255, 0), -1)
        cv2.circle(img, pt_pos4, 2, (0, 255, 0), -1)

        # for i,pt in enumerate(list_points[index]):

        #     pt_pos = (pt[0], pt[1])
        #     cv2.circle(img2, pt_pos, 2, (0, 255, 0), -1)

        
        # cv2.rectangle(img2, (face.left(), face.top()), (face.right(), face.bottom()),
            # (0, 0, 255), 3)


    # cv2.imshow('result', img2)

    
    key = cv2.waitKey(1)    

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    # print(lmList)

    if len(lmList) != 0:
        # fingers = []

        # # Thumb
        # if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
        #     fingers.append(1)
        # else:
        #     fingers.append(0)

        # # 4 Fingers
        # for id in range(1, 5):
        #     if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
        #         fingers.append(1)
        #     else:
        #         fingers.append(0)

        # print(fingers)
        # totalFingers = fingers.count(1)
        # print(totalFingers)
        # total = lmList[5][1] / lmList[17][1]
        # dist = abs(lmList[0][2] - lmList[17][2])
        # dist2 = abs(lmList[5][1] - lmList[17][1])
        
        # cv2.putText(img, "5hand / 17hand: {}".format(total), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # cv2.putText(img, "5hand / 17hand: {}".format(dist2), (10, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        kkc_x = leyex - lmList[8][1] # 검지 눈 거리 
        kkc_y = leyey - lmList[8][2]  

        kkc_dist = math.sqrt((kkc_x ** 2 ) + (kkc_y ** 2)) # 검지, 
        cv2.putText(img, "distance: {}".format(kkc_dist), (10, 90),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 검지끝 눈사이 거리 if
        if kkc_dist < 10:
            
            cv2.putText(img, "finger eye", (10, 120),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


        # 새끼손가락 얼굴안에
        if lmList[tipIds[4]][1] < rex and lmList[tipIds[4]][1] > lex:
            num += 1
            cv2.putText(img, "count: {}".format(num), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if num == 30:                
                h, w, c = overlayList[6].shape
                img[0:h, 0:w] = overlayList[6]
                num = 0
        else:
            num = 0

        # h, w, c = overlayList[totalFingers - 1].shape
        # img[0:h, 0:w] = overlayList[totalFingers - 1]

        # cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
        # cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN,
        #             10, (255, 0, 0), 25)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)

    cv2.imshow("Image", img)     
    cv2.waitKey(1)