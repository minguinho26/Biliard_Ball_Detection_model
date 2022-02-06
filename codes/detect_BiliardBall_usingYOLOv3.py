# YOLOv3 코드(https://github.com/eriklindernoren/PyTorch-YOLOv3)에 있는 YOLOv3에게 당구공 데이터셋을 학습시킨 뒤 테스트하는 코드

# 생성일 : 2022.1.21
# 제작자 : 김민규
from __future__ import division

import numpy as np
import cv2

import torch
import torchvision.transforms as transforms

from codes.YOLOv3_utils import *

# 공의 위치를 나타낼 때 사용하는 클래스
# 당구대에 공이 탐지되지 않을 경우 모든 값을 0.0으로 설정해 화면에 나타낼 수 없게 만들기 위해 제작
# YOLOv3에서 얻은 공의 bbox를 사용할 수 있게 처리 후 BALL에 저장한다
class BALL :
    def __init__(self) :
        # opencv에서 이미지에 도형 그릴 때 필요한 값들
        self.min_x = 0
        self.min_y = 0
        self.max_x = 0
        self.max_y = 0

        # 색깔의 index. YOLOv3의 출력값에 있다.
        self.color_idx = 0
        
    # bbox를 BALL에 저장하는 메소드
    def set_coordi(self, coordi) :
        coordi = coordi.astype(np.int32)

        self.min_x = coordi[0]
        self.min_y = coordi[1]
        self.max_x = coordi[2]
        self.max_y = coordi[3]
        self.color_idx = coordi[5]

    def isValid(self) : 
        return not (self.min_x == 0 & self.min_y == 0 & self.max_x == 0 & self.max_y == 0)

# ====================================
# 전역변수 선언, 정의 =====================
# ====================================
# ball index
global RED_BALL_INDEX 
global WHITE_BALL_INDEX  
global YELLOW_BALL_INDEX 
global MOVING_RED_BALL_INDEX 
global MOVING_WHITE_BALL_INDEX  
global MOVING_YELLOW_BALL_INDEX 

RED_BALL_INDEX = 3
WHITE_BALL_INDEX = 4
YELLOW_BALL_INDEX = 5
MOVING_RED_BALL_INDEX = 6
MOVING_WHITE_BALL_INDEX = 7
MOVING_YELLOW_BALL_INDEX = 8

# 공 색상
global white_color
global yellow_color
global red_color
white_color = (255, 255, 255)
yellow_color = (255,255,0)
red_color = (255, 0, 0)
# ====================================
# 전역변수 선언, 정의 =====================
# ====================================

def detect_biliard_ball(model, image, device, window_name, img_size = 416, nms_thres = 0.1) : # 입력받은 프레임을 가지고 공 탐지

    # 입력받은 프레임을 전처리
    input_img = transforms.Compose([
        DEFAULT_TRANSFORMS,
        Resize(img_size)])(
            (image, np.zeros((1, 5))))[0].unsqueeze(0)
    input_img = input_img.to(device)

    # 전처리한 프레임에서 bbox들을 휙득
    with torch.no_grad():
        detections = model(input_img)
        detections = change_bbox_to_use(detections, nms_thres)
        detections = rescale_boxes(detections[0], img_size, image.shape[:2]).to('cpu')

    # 검출한 bbox들 중 흰공, 노란공, 빨간공이라 판단한 것들을 따로 모음
    red_ball_coordi_group = detections[detections[:,5] == RED_BALL_INDEX]
    white_ball_coordi_group = detections[detections[:,5] == WHITE_BALL_INDEX]
    yellow_ball_coordi_group = detections[detections[:,5] == YELLOW_BALL_INDEX]

    moving_red_ball_coordi_group = detections[detections[:,5] == MOVING_RED_BALL_INDEX]
    moving_white_ball_coordi_group = detections[detections[:,5] == MOVING_WHITE_BALL_INDEX]
    moving_yellow_ball_coordi_group = detections[detections[:,5] == MOVING_YELLOW_BALL_INDEX]

    # 따로 모은 것들에서 가장 confience가 높은 것을 하나씩 뽑음
    # 여기서 얻은 ball_coordi들을 YOLOv3가 탐지한 공들의 좌표로 사용한다
    
    # 공의 위치정보를 담을 BALL 클래스의 객체들을 정의
    white_ball = BALL()
    yellow_ball = BALL()
    red_ball = BALL()
    
    if red_ball_coordi_group.size()[0] > 0 :
        red_ball_coordi = red_ball_coordi_group[red_ball_coordi_group[:,4].argmax(),:].numpy()
    else :
        red_ball_coordi = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    if white_ball_coordi_group.size()[0] > 0 :
        white_ball_coordi = white_ball_coordi_group[white_ball_coordi_group[:,4].argmax(),:].numpy()
    else : 
        white_ball_coordi = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # 만약 공을 탐지하지 못했으면 모든 값을 0.0으로 설정

    if yellow_ball_coordi_group.size()[0] > 0 :
        yellow_ball_coordi = yellow_ball_coordi_group[yellow_ball_coordi_group[:,4].argmax(),:].numpy()
    else :
        yellow_ball_coordi = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # moving ball에서도 추출
    if moving_red_ball_coordi_group.size()[0] > 0 :
        moving_red_ball_coordi = moving_red_ball_coordi_group[moving_red_ball_coordi_group[:,4].argmax(),:].numpy()
    else :
        moving_red_ball_coordi = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    if moving_white_ball_coordi_group.size()[0] > 0 :
        moving_white_ball_coordi = moving_white_ball_coordi_group[moving_white_ball_coordi_group[:,4].argmax(),:].numpy()
    else : 
        moving_white_ball_coordi = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # 만약 공을 탐지하지 못했으면 모든 값을 0.0으로 설정
        
    if moving_yellow_ball_coordi_group.size()[0] > 0 :
        moving_yellow_ball_coordi = moving_yellow_ball_coordi_group[moving_yellow_ball_coordi_group[:,4].argmax(),:].numpy()
    else :
        moving_yellow_ball_coordi = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    
    # confidence score보고 moving ball과 ball중 하나를 공의 좌표로 선정
    if moving_red_ball_coordi[4] > red_ball_coordi[4] :
        red_ball.set_coordi(moving_red_ball_coordi)
    else :
        red_ball.set_coordi(red_ball_coordi)

    if moving_white_ball_coordi[4] > white_ball_coordi[4] :
        white_ball.set_coordi(moving_white_ball_coordi)
    else :
        white_ball.set_coordi(white_ball_coordi)

    if moving_yellow_ball_coordi[4] > yellow_ball_coordi[4] :
        yellow_ball.set_coordi(moving_yellow_ball_coordi)
    else :
        yellow_ball.set_coordi(yellow_ball_coordi)

    detections = [white_ball, yellow_ball, red_ball] # ball 객체들로 구성된 리스트 생성

    for detect_bbox in detections :
        # 공의 종류에 맞는 색을 가진 bbox를 출력 
        if detect_bbox.color_idx == RED_BALL_INDEX or detect_bbox.color_idx == MOVING_RED_BALL_INDEX : # 빨간공
            if detect_bbox.isValid() == True : # Ball이 가지고 있는 값들이 모두 0.0이 아니면 공의 좌표를 출력. 앞서 공을 탐지 못했을시 모든 값을 0.0으로 설정한 이유
                cv2.rectangle(image, (detect_bbox.min_x, detect_bbox.min_y), (detect_bbox.max_x, detect_bbox.max_y), red_color, 2)
        elif detect_bbox.color_idx == WHITE_BALL_INDEX or detect_bbox.color_idx == MOVING_WHITE_BALL_INDEX : # 흰공
            if detect_bbox.isValid() == True :
                cv2.rectangle(image, (detect_bbox.min_x, detect_bbox.min_y), (detect_bbox.max_x, detect_bbox.max_y), white_color, 2)
        elif detect_bbox.color_idx == YELLOW_BALL_INDEX or detect_bbox.color_idx == MOVING_YELLOW_BALL_INDEX : # 노란공
            if detect_bbox.isValid() == True :
                cv2.rectangle(image, (detect_bbox.min_x, detect_bbox.min_y), (detect_bbox.max_x, detect_bbox.max_y), yellow_color, 2)
        
    cv2.imshow(window_name, image[:, :, [2, 1, 0]]) # RGB -> BGR변환. opencv는 BGR을 사용한다
    cv2.waitKey(1)
