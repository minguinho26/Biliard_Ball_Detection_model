# YOLOv3 코드(https://github.com/eriklindernoren/PyTorch-YOLOv3)에 있는 YOLOv3에게 당구공 데이터셋을 학습시킨 뒤 테스트하는 코드

# 생성일 : 2022.1.21
# 제작자 : 김민규
from __future__ import division

import numpy as np
import cv2
import timeit
import copy

import torch
import torchvision.transforms as transforms

from codes.YOLOv3_utils import *
from ball_detect_program import *

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

RED_BALL_INDEX           = 4
WHITE_BALL_INDEX         = 5
YELLOW_BALL_INDEX        = 6
MOVING_RED_BALL_INDEX    = 7
MOVING_WHITE_BALL_INDEX  = 8
MOVING_YELLOW_BALL_INDEX = 9
# ====================================
# 전역변수 선언, 정의 =====================
# ====================================
def detect_biliard_ball(model, image, device, window_name, BALLS, img_size = 416, nms_thres = 0.5) : # 입력받은 프레임을 가지고 공 탐지

    time_start = timeit.default_timer() # start time

    ori_image = image.copy() # 공을 그리기 전의 이미지

    # 공들의 좌표를 출력하기 위해 사용하는 BALL 클래스의 객체들
    white_ball = BALLS[0]
    yellow_ball = BALLS[1]
    red_ball = BALLS[2]
    
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

    detections = [red_ball, white_ball, yellow_ball]

    for detect_bbox in detections :
        # 공의 종류에 맞는 색을 가진 bbox를 출력 
        image = detect_bbox.print_coordi(copy.deepcopy(image))
    
    time_end = timeit.default_timer() # end time 
    FPS = int(1./(time_end - time_start ))

    cv2.putText(image,"FPS : " + str(FPS), (1090, 43), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2)

    cv2.imshow(window_name, image[:, :, [2, 1, 0]]) # RGB -> BGR변환. opencv는 BGR을 사용한다