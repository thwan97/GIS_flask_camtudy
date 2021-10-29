# main_views.py

from flask import Blueprint, render_template, Response
from werkzeug.utils import redirect

bp = Blueprint('main', __name__, url_prefix='/')


@bp.route('/hello')
def hello_pybo():
    return 'Hello, Pybo!'


# main_page
from .func import preview_camera
 # 메인페이지로 라우팅 (페이지) 
@bp.route('/')
def index():
    return render_template('main_page.html')
# 메인에서 미리보기 카메라 켜지는 부분 (기능)
@bp.route('/preview_camera')
def preview_camera_route():
    return Response(preview_camera.preview_camera(), mimetype='multipart/x-mixed-replace; boundary=frame')


# htm_page
from .func import htm_camera
# 메인 페이지에서 스터디 들어가기 버튼 클릭 -> htm_page로 감 (여기서 htm_camera를 킴) (페이지)
@bp.route('/htm_page')
def htm_page_route():
    return render_template('htm_page.html')
@bp.route('/htm_camera')
def htm_camera_route():
    return Response(htm_camera.htm_camera(), mimetype='multipart/x-mixed-replace; boundary=img')
