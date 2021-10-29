import cv2 
import dlib 
from scipy.spatial import distance
from imutils import face_utils
from scipy.spatial import distance

#hand
import HandTrackingModule as htm


cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

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

#눈, 손가락끝거리
def dist_eye_finger(lmList, face_landmarks, val1, val2):
    x = abs(face_landmarks.part(val1).x - lmList[val2][1])
    y = abs(face_landmarks.part(val1).y - lmList[val2][2])
    return (x ** 2 + y ** 2) ** (1/2)

count = 0
count2 = 0
count3 = 0
total = 0
first = 0
height = -1
width = -1
while True:
    success,img = cap.read()
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = detector(imgGray)
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
            first += 1
        else:
            height = face_size(27, 8, face_landmarks) ## 화면속 세로 크기
            first += 1
    
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
        
        dist_1 = dist_eye_finger(lmList, face_landmarks, 40, 8) #왼쪽눈 검지 거리
        dist_2 = dist_eye_finger(lmList, face_landmarks, 47, 8) #오른쪽눈 검지 거리
        if dist_1 < 15 or dist_2 < 15:
            count3 += 1
            if count3 == 30:
                print("눈비빔")
                count3 = 0
        else:
            count3 = 0
        
        print(dist_1)


        
    if first > 1:
        if (height - fis_height + width - fis_width) > 50: ## 첫 화면 얼굴 크기보다 나중 화면 얼굴 크기가 일정 값 이상 커질때
            cv2.putText(img, "go back!", (100, 100),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)    
    cv2.putText(img, "Blink Count: {}".format(total), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # if (height > 0) and (width > 0):
    #     print(fis_height, fis_width, height, width)
    cv2.imshow('Video',img)
    if cv2.waitKey(1) & 0xff==ord('q'):
        break
