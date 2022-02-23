# 촬영중인 카메라에서 얻은 프레임에 있는 당구공 위치를 YOLOv3으로 탐지하는 프로그램

# 생성일 : 2022.1.21
# 제작자 : 김민규

from tkinter import scrolledtext

from cv2 import circle
from codes.YOLOv3_models import *
from codes.detect_BiliardBall_usingYOLOv3 import *
import numpy as np
import os
import cv2

# 이미지 저장에 사용
import datetime
from PIL import Image

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
        self.coordi_list = [] # 공의 좌표를 기록하는 곳
        # 색깔의 index
        self.color_idx = color_idx
        self.status = STATUS['NOTHING']
        self.isMoving_oneTIME = False
        self.movingStartTime = 0
        
    # 마스크를 만드는 함수. 이걸로 만든 마스크로 특정 조건을 만족하는 픽셀들을 제거한다
    def set_mask(self, mask, FILTERS) :

        for filter in FILTERS :
            mask[filter] = 0
        return mask

    # bbox를 BALL에 저장하는 메소드
    def set_coordi(self, image, coordi, FirstMovingBall_COLOR, ball_crush_stack) :
        isSetCOORDI = False # 현재 프레임에서 공을 탐지했는가?

        # YOLO가 공이 포함된 bbox를 탐지했으면
        if np.sum(np.abs(coordi)) != 0 :
            coordi[coordi[:] < 0] = 0.0
            ball_with_bbox = image[np.floor(coordi[1]).astype(np.int32):np.ceil(coordi[3]).astype(np.int32), np.floor(coordi[0]).astype(np.int32):np.ceil(coordi[2]).astype(np.int32),:] 
            ball_with_bbox = ball_with_bbox[:, :, [2, 1, 0]]

            bbox_w = ball_with_bbox.shape[1]
            bbox_h = ball_with_bbox.shape[0]

            # hsv로 변경 후 빨간공, 흰공, 노란공을 탐지
            ball_with_bbox_hsv = cv2.cvtColor(ball_with_bbox, cv2.COLOR_BGR2HSV)

            mask = np.ones((bbox_h, bbox_w ))
            max_val_RGB = np.argmax(ball_with_bbox, axis=2) # max value index of RGB of pixel in bbox

            # 원하는 색상의 공을 찾기 위한 필터링을 실시=======================================================
            if self.color_idx == BALL_COLOR['RED'] :
                filter_channel_max_pixel = np.where((max_val_RGB[:,:] == 0)) # 파란색이 제일 높은값을 가지는 픽셀을 제외
                
                remove_whiteball = np.where((ball_with_bbox_hsv[:,:,1] < 80)) # 흰색의 saturation을 검출 후 제거
                remove_yellowball = np.where((ball_with_bbox_hsv[:,:,0] > 15) & (ball_with_bbox_hsv[:,:,0] < 165)) # 노란색과 흰색은 hue에서 차이를 보인다. 이점을 이용해 노란색을 검출한다
                mask = self.set_mask(mask, [filter_channel_max_pixel, remove_whiteball, remove_yellowball]) # 빨간공을 제외한 나머지 것들을 제거하기 위한 마스크를 생성
                
                ball_with_bbox_grayscale = cv2.cvtColor(ball_with_bbox, cv2.COLOR_BGR2GRAY) # 흑백 이미지로 변환 후 mask로 빨간공, 노란공 등을 제거
                # 특정 조건을 만족하는 픽셀들을 제거
                ball_with_bbox_grayscale[mask < 1] = 0
                ball_with_bbox_grayscale[ball_with_bbox_grayscale > 250] = 0
                ball_with_bbox_grayscale[ball_with_bbox_grayscale < 120] = 0

            elif self.color_idx == BALL_COLOR['WHITE'] :
                
                filter_whiteball = np.where((ball_with_bbox_hsv[:,:,1] > 60))

                mask = self.set_mask(mask, [filter_whiteball])
                ball_with_bbox_grayscale = cv2.cvtColor(ball_with_bbox, cv2.COLOR_BGR2GRAY)
                ball_with_bbox_grayscale[mask < 1] = 0
                ball_with_bbox_grayscale[ball_with_bbox_grayscale[:,:] < 250] = 0
                
            elif self.color_idx == BALL_COLOR['YELLOW'] :
                remove_whiteball = np.where((ball_with_bbox_hsv[:,:,1] < 80))
                remove_redball = np.where((ball_with_bbox_hsv[:,:,0] <= 15) | (ball_with_bbox_hsv[:,:,0] >= 165))
                remove_biliardfield = np.where((ball_with_bbox_hsv[:,:,2] < 200))

                mask = self.set_mask(mask, [remove_whiteball, remove_redball, remove_biliardfield])
                ball_with_bbox_grayscale = cv2.cvtColor(ball_with_bbox, cv2.COLOR_BGR2GRAY)
                ball_with_bbox_grayscale[mask < 1] = 0
            # 원하는 색상의 공을 찾기 위한 필터링을 실시=======================================================

            isSetCOORDI = False
            temp_coordi = None
            base_coordi = (coordi[0].astype(np.int32), coordi[1].astype(np.int32))# 전체 이미지에서 얻은 bbox의 left-up 꼭지점 좌표
            
            # 덩어리가 뭉친 곳을 공이라 판정
            # 원래 HoughCircles()를 사용하여 공을 찾았는데 공의 모양대로 걸러지는 경우가 별로 없어서 '살아남은 픽셀'들을 감싼 폐곡선(contour)을 탐지, 폐곡선이 둘러싼 영역의 넓이가 특정값을 넘기면 이를 공으로 판단한다.
            _,threshold = cv2.threshold(ball_with_bbox_grayscale, 110, 255, 
                            cv2.THRESH_BINARY)
            contours,_=cv2.findContours(threshold, cv2.RETR_TREE,
                            cv2.CHAIN_APPROX_SIMPLE)
                
            for cnt in contours :
                area = cv2.contourArea(cnt)

                if area > 300:    
                    # 무게중심 계산                   
                    M = cv2.moments(cnt)
                    cX = int(M['m10'] / M['m00'])
                    cY = int(M['m01'] / M['m00'])
                    radius = np.sqrt(area/ 2).astype(np.int32)
                    
                    # 원이 너무 크면 조정 
                    if radius > 16 : 
                        radius = 16

                    # 좌표로 설정
                    temp_coordi = COORDI(base_coordi[0] + cX, base_coordi[1] + cY, radius)
            
            # 공을 탐지했으면
            if temp_coordi != None :             
                if len(self.coordi_list) >= 1 :
                    # 공이 처음으로 움직였으면
                    if self.isMoving_oneTIME == False :
                        diff_with_initCOORDI_x = self.coordi_list[0].x - temp_coordi.x
                        diff_with_initCOORDI_y = self.coordi_list[0].y - temp_coordi.y

                        distance_with_initCOORDI = np.sqrt(diff_with_initCOORDI_x * diff_with_initCOORDI_x + diff_with_initCOORDI_y * diff_with_initCOORDI_y)

                        if distance_with_initCOORDI > 5.0 : # 지금까지 움직였던 거리가 일정한 값을 만족하면 '공이 움직이기 시작했다'고 판단한다 
                            self.status = STATUS['MOVING'] # 공의 상태를 변경
                            self.movingStartTime = datetime.datetime.now() # 움직이기 시작한 시간
                            # 내가 친 공은 맨처음 움직인 공이 된다. 
                            if FirstMovingBall_COLOR == None :
                                FirstMovingBall_COLOR = self.color_idx
                                ball_crush_stack[FirstMovingBall_COLOR] = 1
                            self.isMoving_oneTIME = True
                        else : 
                            self.status = STATUS['WAITING'] # 일정값을 만족하지 않으면 움직이지 않는다고 판단
                    # 움직였던 공이었으면
                    else :
                        diff_with_preCOORDI_x = self.coordi_list[-1].x - temp_coordi.x
                        diff_with_preCOORDI_y = self.coordi_list[-1].y - temp_coordi.y

                        distance_with_preCOORDI = np.sqrt(diff_with_preCOORDI_x * diff_with_preCOORDI_x + diff_with_preCOORDI_y * diff_with_preCOORDI_y)

                        # 계속 움직이는 공은 완전히 멈출 때까지 움직인다고 판단
                        if float(distance_with_preCOORDI) > 2.0 : 
                            self.status = STATUS['MOVING']
                        else :
                            self.status = STATUS['WAITING'] 
                    self.coordi_list.append(temp_coordi)
                else : 
                    self.coordi_list.append(temp_coordi)
            # 공을 탐지하지 못했을 때
            else : 
                self.status = STATUS['NOTHING']
                if len(self.coordi_list) > 1 :
                    del self.coordi_list[:-1]
        # YOLO가 bbox를 탐지하지 못했을 경우
        else :
            self.status = STATUS['NOTHING']
            if len(self.coordi_list) > 1 :
                del self.coordi_list[:-1]
                
        return isSetCOORDI, FirstMovingBall_COLOR, ball_crush_stack

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
    red_ball = BALL(BALL_COLOR['RED'])
    white_ball = BALL(BALL_COLOR['WHITE'])
    yellow_ball = BALL(BALL_COLOR['YELLOW'])
    

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
        
        fraem_copied = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        _,threshold = cv2.threshold(fraem_copied, 110, 255, cv2.THRESH_BINARY)
        contours,_=cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours :
            area = cv2.contourArea(cnt)
            if area > 400: 
                for cnt in contours :
                    area = cv2.contourArea(cnt)
                    epsilon = 0.02 * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)
                    cv2.drawContours(fraem_copied,[approx],0,(0,0,0),5)
                    cv2.imshow('test_frame', fraem_copied)

        if not status:
            break
        elif cv2.waitKey(1) == ord('q') : # q 누르면 탈출
            break

        # numpy와 pytorch는 RGB를 사용하는데 opencv는 BGR을 사용하기 때문에 변환한 값을 넣어준다
        score, FirstMovingBall_COLOR, Waiting_start_time, ball_crush_stack = detect_biliard_ball(model, frame.copy()[:, :, [2, 1, 0]], device, 'Detect_ball', [red_ball, white_ball, yellow_ball], score, FirstMovingBall_COLOR, Waiting_start_time, ball_crush_stack) # 공 탐지 후 화면에 보여줌
    capture.release() # 캡처 중지
    cv2.destroyAllWindows() # 화면 생성 중단

if __name__ == "__main__":
    main()
