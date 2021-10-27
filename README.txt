
ㅁ 2021.10.21
- MediaPipe 라이브러리를 이용한 HandTrackingModule(입에 손가락, 얼굴에 손, 눈비빔) + Pose(만세!)를
app.py에 적용시킴 (but, 웹캠이 나오지 않음)
- 웹캠이 나오게 preview_camera를 틀고 htm_camera를 같이 틀면 서버가 터지는 현상. (둘중 하나만 키면 가능)

< v2 >
ㅁ 2021.10.27
-           ret, buffer = cv2.imencode('.jpg', img)
            img = buffer.tobytes()
            yield (b'--img\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')
웹캠 트는 부분에서 frame -> img로 변수 이름 변경 해줬더니 웹캠이 틀어짐

ㅁ 추가할 사항
- 나의 상태 (ex) 얼굴안에 손, 눈비빔, 만세 등) 이 consol뿐만 아니라 html에 표현하기
- 모듈화 하기