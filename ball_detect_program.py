# 촬영중인 카메라에서 얻은 프레임에 있는 당구공 위치를 YOLOv3으로 탐지하는 프로그램

# 생성일 : 2022.1.21
# 제작자 : 김민규

from codes.YOLOv3_models import *
from codes.detect_BiliardBall_usingYOLOv3 import *
import numpy as np
import os
import cv2

# 이미지 저장에 사용
import datetime

# NOTHING : 아직 상태가 안정해짐 or 현재 프레임에서 공을 탐지하지 못함, WAITING : 가만히 있는 상태, MOVING : 공이 움직이는 상태
global STATUS
STATUS = {'NOTHING' : 0, 'WAITING' : 1, 'MOVING' : 2}
BALL_COLOR = {'RED' : 0, 'WHITE' : 1, 'YELLOW' : 2}

# 공의 위치를 나타낼 때 사용하는 클래스
# 22.2.17에 bbox의 점 대신 중심좌표를 저장하게끔 설정
class COORDI :
    def __init__(self, x_, y_, r_) :
        self.x = x_
        self.y = y_
        self.r = r_
    def get_center(self) : 
        return (self.x, self.y)
        
# 카메라에 탐지되는 공의 최근 10개 좌표를 저장하는 클래스
class BALL :
    def __init__(self, color_idx) :
        # opencv에서 이미지에 도형 그릴 때 필요한 값들
        self.coordi_list = []
        # 색깔의 index
        self.color_idx = color_idx
        self.status = STATUS['NOTHING']
        self.isMoving_oneTIME = False
        self.movingStartTime = 0
        
    # bbox를 BALL에 저장하는 메소드
    # extract bbox in image and use HoughCircles() to get more precise ball's coordinate and radius
    def set_coordi(self, image, coordi, isHit) :
        isSetCOORDI = False # 현재 프레임에서 공을 탐지했는가?
        if np.sum(np.abs(coordi)) != 0 :
            ball_with_bbox = image[np.floor(coordi[1]).astype(np.int32):np.ceil(coordi[3]).astype(np.int32), np.floor(coordi[0]).astype(np.int32):np.ceil(coordi[2]).astype(np.int32),:] 
            ball_with_bbox_grayscale = cv2.GaussianBlur(cv2.cvtColor(ball_with_bbox, cv2.COLOR_BGR2GRAY), (0, 0), 1) 
            circles = cv2.HoughCircles(ball_with_bbox_grayscale, cv2.HOUGH_GRADIENT, 1, 10, param1 = 250, param2 = 10, minRadius = 13, maxRadius = 18)  

            isSetCOORDI = True

            temp_coordi = None
            base_coordi = (coordi[0], coordi[1])# 전체 이미지에서 얻은 bbbox의 left-up 꼭지점 좌표
            if type(circles) != type(None) :
                # print(circles[0].shape[0])
                for i in circles[0]:
                    temp_coordi = COORDI(int(base_coordi[0] + i[0]), int(base_coordi[1] + i[1]), np.ceil(i[2]).astype(np.int32))
                    
            if temp_coordi == None :
                width = coordi[2] - coordi[0]
                height = coordi[3] - coordi[1]
                temp_coordi = COORDI(int(base_coordi[0] + width/2), int(base_coordi[1] + height/2), np.ceil(min(width, height)/2).astype(np.int32))
            if len(self.coordi_list) >= 1 :
                diff_with_preCOORDI_x = self.coordi_list[-1].x - temp_coordi.x
                diff_with_preCOORDI_y = self.coordi_list[-1].y - temp_coordi.y

                distance_with_preCOORDI = np.sqrt(diff_with_preCOORDI_x * diff_with_preCOORDI_x + diff_with_preCOORDI_y * diff_with_preCOORDI_y)

                if self.isMoving_oneTIME == False :
                    if float(distance_with_preCOORDI) > 8.0 :
                        self.status = STATUS['MOVING']
                        self.movingStartTime = datetime.datetime.now()
                        self.isMoving_oneTIME = True
                    else : 
                        self.status = STATUS['WAITING']
                else :
                    if float(distance_with_preCOORDI) > 1.0 : 
                        self.status = STATUS['MOVING']
                    else :
                        self.status = STATUS['WAITING'] 
                self.coordi_list.append(temp_coordi)
            else : 
                self.coordi_list.append(temp_coordi)
        else :
            self.status = STATUS['NOTHING']
            if len(self.coordi_list) > 0 :
                self.coordi_list.pop(0)
                
        return isSetCOORDI

# YOLOv3을 불러오기 위한 메소드
def ready_for_detect() :
    # 모델의 구조가 담긴 cfg와 학습시킨 가중치들이 담긴 pth의 경로
    file_location = os.path.dirname(os.path.abspath(__file__))
    model_path = file_location + '/yolov3/yolov3-for-Biliard-10Classes.cfg'
    pretrained_weights = file_location + '/yolov3/yolov3_weights_10Classes.pth'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 파이토치를 이용해 만든 객체, 변수들을 gpu에서 돌릴지 cpu에서 돌릴지 결정.
    model = load_model(model_path, pretrained_weights) # 학습시킨 가중치가 저장된 YOLOv3을 불러옴
    model.eval() # YOLOv3을 테스트 모드로 변경
    return model, device

def main() :
    model, device = ready_for_detect() # YOLOv3과 파이토치 기반 객체, 변수를 돌릴 장치(cpu or gpu)에 대한 정보를 휙득

    capture = cv2.VideoCapture(0) # 연결된 카메라 혹은 비디오에서 프레임을 얻기 위해 필요한 객체. 0을 매개변수로 전달하면 연결된 카메라에서 프레임을 얻어온다

    # 캡처하는 프레임의 해상도 조정
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # 공의 위치정보를 담을 BALL 클래스의 객체들을 정의
    white_ball = BALL(BALL_COLOR['WHITE'])
    yellow_ball = BALL(BALL_COLOR['YELLOW'])
    red_ball = BALL(BALL_COLOR['RED'])

    score = 0
    FirstMovingBall_COLOR = None
    Waiting_start_time = None

    ball_crush_stack = [0, 0, 0] # 각 공을 큐대로 쳤을 때 다른 공과 충돌한 횟수. 빨간공, 흰공, 노란공 순서

    # 카메라로 프레임을 캡처하지 못했으면 프로그램 종료
    if not capture.isOpened():
        print("Could not open webcam")
        exit()

    while True:
        # 찍고있는 카메라에서 캡처 상태와 찍고있는 프레임 얻기
        status, frame = capture.read() # qstatus는 boolean이고 frame은 numpy array다.

        if not status:
            break
        elif cv2.waitKey(1) == ord('q') : # q 누르면 탈출
            break
        frame = frame[:, :, [2, 1, 0]] # # BGR -> RGB. numpy와 pytorch는 RGB를 사용한다 
        frame, score, FirstMovingBall_COLOR, Waiting_start_time, ball_crush_stack = detect_biliard_ball(model, frame.copy(), device, 'Detect_ball', [white_ball, yellow_ball, red_ball], score, FirstMovingBall_COLOR, Waiting_start_time, ball_crush_stack) # 공 탐지 후 화면에 보여줌
    capture.release() # 캡처 중지
    cv2.destroyAllWindows() # 화면 생성 중단

if __name__ == "__main__":
    main()
