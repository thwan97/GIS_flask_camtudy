

from flask import Flask, render_template, Response


# 객체 생성 부분
app=Flask(__name__)



# main_page
import views.preview_camera as preview_camera
 # 메인페이지로 라우팅 (페이지) 
@app.route('/')
def main_page_route():
    return render_template('main_page.html')

# 메인에서 미리보기 카메라 켜지는 부분 (기능)
@app.route('/preview_camera')
def preview_camera_route():
    return Response(preview_camera.preview_camera(), mimetype='multipart/x-mixed-replace; boundary=frame')



# htm_page
import views.htm_camera as htm_camera
# 메인 페이지에서 스터디 들어가기 버튼 클릭 -> htm_page로 감 (여기서 htm_camera를 킴) (페이지)
@app.route('/htm_page')
def htm_page_route():
    return render_template('htm_page.html')
@app.route('/htm_camera')
def htm_camera_route():
    return Response(htm_camera.htm_camera(), mimetype='multipart/x-mixed-replace; boundary=img')



if __name__=='__main__':
    app.run(debug=True)