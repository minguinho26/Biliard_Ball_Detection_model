# YOLOv3 코드(https://github.com/eriklindernoren/PyTorch-YOLOv3)에 있는 YOLOv3에게 당구공 데이터셋을 학습시킨 뒤 테스트하는 코드

# 생성일 : 2022.1.21
# 제작자 : 김민규
from __future__ import division

import numpy as np
import cv2
import timeit
import datetime

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

# 당구대에서 탐지된 공들 중 넓이가 일정 수치보다 높은 공을 걸러내는 함수
# 하얀 접시가 공으로 판정되는 일이 발생해 이러한 일을 방지하고자 사용
def filter_in_BiliardField(detect_ball_coordi_group, threashold_area) :

    width_group = detect_ball_coordi_group[:,2] - detect_ball_coordi_group[:,0]
    height_group = detect_ball_coordi_group[:,3] - detect_ball_coordi_group[:,1]

    area_group = torch.multiply(width_group, height_group)

    detect_ball_coordi_group = detect_ball_coordi_group[area_group[:] <= threashold_area]

    return detect_ball_coordi_group

# 두 공 사이의 거리를 측정하는 함수
def get_distance(ball1, ball2) : 

    point1 = ball1.coordi_list[-1].get_center()
    point2 = ball2.coordi_list[-1].get_center()

    diff_x = abs(point1[0] - point2[0])
    diff_y = abs(point1[1] - point2[1])

    return np.sqrt(diff_x * diff_x + diff_y * diff_y)

# 내가 친 공이 나머지 두 공을 맞췄는지 판정하는 함수
def determine_crush_stack(ball_crush_stack, FirstMovingBall_COLOR, BALLS, now) :
    target_color_idx = FirstMovingBall_COLOR# crush_stack을 올릴 공의 index
    is_Stack_value = 0

    red_ball = BALLS[0]
    white_ball = BALLS[1]
    yellow_ball = BALLS[2]

    target_ball_1 = None
    target_ball_2 = None

    if target_color_idx == BALL_COLOR['RED'] :
        target_ball_1 = white_ball
        target_ball_2 = yellow_ball
    elif target_color_idx == BALL_COLOR['WHITE'] :
        target_ball_1 = yellow_ball
        target_ball_2 = red_ball
    elif target_color_idx == BALL_COLOR['YELLOW'] :
        target_ball_1 = red_ball
        target_ball_2 = white_ball

    
    if ball_crush_stack[target_color_idx] == 0 :
        if type(target_ball_1.movingStartTime) == type(0) and type(target_ball_2.movingStartTime) == type(0) :
            is_Stack_value = 0
        else : 
            if type(target_ball_1.movingStartTime) == type(now) and type(target_ball_2.movingStartTime) == type(now) : # 한 번에 두개의 공을 침
                is_Stack_value = 2
            else : 
                is_Stack_value = 1
    elif ball_crush_stack[target_color_idx] == 1 :
        if type(target_ball_1.movingStartTime) == type(now) and type(target_ball_2.movingStartTime) == type(now) :
            is_Stack_value = 1

    ball_crush_stack[target_color_idx] += is_Stack_value

    return ball_crush_stack


def detect_biliard_ball(model, image, device, window_name, BALLS, score, FirstMovingBall_COLOR, Waiting_start_time, ball_crush_stack, img_size = 416) : # 입력받은 프레임을 가지고 공 탐지

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
        detections = change_bbox_to_use(detections)
        detections = rescale_boxes(detections, img_size, image.shape[:2]).to('cpu')

    # 검출한 bbox들 중 흰공, 노란공, 빨간공이라 판단한 것들을 따로 모음
    red_ball_coordi_group = detections[detections[:,5] == RED_BALL_INDEX]
    white_ball_coordi_group = detections[detections[:,5] == WHITE_BALL_INDEX]
    yellow_ball_coordi_group = detections[detections[:,5] == YELLOW_BALL_INDEX]

    moving_red_ball_coordi_group = detections[detections[:,5] == MOVING_RED_BALL_INDEX]
    moving_white_ball_coordi_group = detections[detections[:,5] == MOVING_WHITE_BALL_INDEX]
    moving_yellow_ball_coordi_group = detections[detections[:,5] == MOVING_YELLOW_BALL_INDEX]

    # 특정 조건을 만족하는 bbox만 사용
    threashold_area = 3000.0

    if red_ball_coordi_group.size()[0] > 0 :
        red_ball_coordi_group = filter_in_BiliardField(red_ball_coordi_group, threashold_area)
    if white_ball_coordi_group.size()[0] > 0 :
        white_ball_coordi_group = filter_in_BiliardField(white_ball_coordi_group, threashold_area)
    if yellow_ball_coordi_group.size()[0] > 0 :
        yellow_ball_coordi_group = filter_in_BiliardField(yellow_ball_coordi_group, threashold_area)
        
    if moving_red_ball_coordi_group.size()[0] > 0 :
        moving_red_ball_coordi_group = filter_in_BiliardField(moving_red_ball_coordi_group, threashold_area)
    if moving_white_ball_coordi_group.size()[0] > 0 :
        moving_white_ball_coordi_group = filter_in_BiliardField(moving_white_ball_coordi_group, threashold_area)
    if moving_yellow_ball_coordi_group.size()[0] > 0 :
        moving_yellow_ball_coordi_group = filter_in_BiliardField(moving_yellow_ball_coordi_group, threashold_area)

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
    isHit = False
    if Waiting_start_time != None : 
        isHit = True
    if moving_red_ball_coordi[4] > red_ball_coordi[4] :
        target_bbox = moving_red_ball_coordi
    else :
        target_bbox = red_ball_coordi
    isSETCOORDI_RED = red_ball.set_coordi(image.copy(), target_bbox, isHit)

    if moving_white_ball_coordi[4] > white_ball_coordi[4] :
        target_bbox = moving_white_ball_coordi
    else :
        target_bbox = white_ball_coordi
    isSETCOORDI_WHITE = white_ball.set_coordi(image.copy(), target_bbox, isHit)


    if moving_yellow_ball_coordi[4] > yellow_ball_coordi[4] :
        target_bbox = moving_yellow_ball_coordi
    else :
        target_bbox = yellow_ball_coordi
    isSETCOORDI_YELLOW = yellow_ball.set_coordi(image.copy(), target_bbox, isHit)

    # set first moving ball
    now = datetime.datetime.now()
    if type(red_ball.movingStartTime) == type(now) and type(white_ball.movingStartTime) == type(0) and type(yellow_ball.movingStartTime) == type(0) :
        FirstMovingBall_COLOR = BALL_COLOR['RED']
    elif type(red_ball.movingStartTime) == type(0) and type(white_ball.movingStartTime) == type(now) and type(yellow_ball.movingStartTime) == type(0) :
        FirstMovingBall_COLOR = BALL_COLOR['WHITE']
    elif type(red_ball.movingStartTime) == type(0) and type(white_ball.movingStartTime) == type(0) and type(yellow_ball.movingStartTime) == type(now) :
        FirstMovingBall_COLOR = BALL_COLOR['YELLOW']
    
    if FirstMovingBall_COLOR != None : 
        ball_crush_stack = determine_crush_stack(ball_crush_stack, FirstMovingBall_COLOR, BALLS, now)

    if red_ball.status == STATUS['WAITING'] and  white_ball.status == STATUS['WAITING'] and yellow_ball.status == STATUS['WAITING'] and FirstMovingBall_COLOR != None and len(red_ball.coordi_list) > 1 and len(white_ball.coordi_list) > 1 and len(yellow_ball.coordi_list) > 1:
        if red_ball.isMoving_oneTIME == True and white_ball.isMoving_oneTIME == True and yellow_ball.isMoving_oneTIME == True : 
                score +=1
        
        if Waiting_start_time == None :
            Waiting_start_time = datetime.datetime.now()
                    
    elif FirstMovingBall_COLOR == None :
        if len(red_ball.coordi_list) > 10 :
            del red_ball.coordi_list[:-10]
        if len(white_ball.coordi_list) > 10 :    
            del white_ball.coordi_list[:-10]
        if len(yellow_ball.coordi_list) > 10 :    
            del yellow_ball.coordi_list[:-10]

    if Waiting_start_time != None and (now - Waiting_start_time).total_seconds() > 3.0 : 
        red_ball.isMoving_oneTIME    = False 
        white_ball.isMoving_oneTIME  = False 
        yellow_ball.isMoving_oneTIME = False

        if len(red_ball.coordi_list) > 1 :
            del red_ball.coordi_list[:-1]
        if len(white_ball.coordi_list) > 1 :    
            del white_ball.coordi_list[:-1]
        if len(yellow_ball.coordi_list) > 1 :    
            del yellow_ball.coordi_list[:-1]

        red_ball.movingStartTime = 0
        white_ball.movingStartTime = 0
        yellow_ball.movingStartTime = 0
        ball_crush_stack = [0, 0, 0]
        FirstMovingBall_COLOR = None
        Waiting_start_time = None

    detections = [red_ball, white_ball, yellow_ball]

    for detect_bbox in detections :
        # 공의 종류에 맞는 색을 가진 bbox를 출력
        if detect_bbox.color_idx == BALL_COLOR['RED'] :
            color = (255, 82, 41)
        elif detect_bbox.color_idx == BALL_COLOR['WHITE'] :
            color = (230, 230, 230)
        elif detect_bbox.color_idx == BALL_COLOR['YELLOW'] :
            color = (230,204,0)
        
        if len(detect_bbox.coordi_list) > 1 and FirstMovingBall_COLOR != None :
            for i in range(1, len(detect_bbox.coordi_list)) : 
                cv2.arrowedLine(image, detect_bbox.coordi_list[i - 1].get_center(), detect_bbox.coordi_list[i].get_center(), color, 3, tipLength=0.2, line_type=4)
        
        # print last coordinate of ball as more bigger circle
        # I think there are some bugs in this area(22.02.16)
        # first_moving_ball 판단하는 코드에 버그가 있는듯 하다. 
        if len(detect_bbox.coordi_list) > 0 :
            print(detect_bbox.coordi_list[-1].r)
            cv2.circle(image, detect_bbox.coordi_list[-1].get_center(), detect_bbox.coordi_list[-1].r, color, -1)
        
    time_end = timeit.default_timer() # end time 
    FPS = int(1./(time_end - time_start ))

    print_str = "FPS : " + str(FPS) + ", SCORE : " + str(score)
    
    cv2.putText(image, print_str, (900, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2)

    if red_ball.status != STATUS['NOTHING'] and  white_ball.status != STATUS['NOTHING'] and yellow_ball.status != STATUS['NOTHING'] :
        
        distance_red_with_white = get_distance(red_ball, white_ball)
        distance_white_with_yellow = get_distance(white_ball, yellow_ball)
        distance_yellow_with_red = get_distance(yellow_ball, red_ball)

        distance_rw_str = "distance_red_white : " + str(distance_red_with_white)
        distance_wy_str = "distance_white_yellow : " + str(distance_white_with_yellow)
        distance_yr_str = "distance_yellow_red : " + str(distance_yellow_with_red)

        ball_crush_stack_print = "ball_crush_stack : " + str(ball_crush_stack)

        cv2.putText(image, distance_rw_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2)
        cv2.putText(image, distance_wy_str, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2)
        cv2.putText(image, distance_yr_str, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2)
        cv2.putText(image, ball_crush_stack_print, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2)

        

    if FirstMovingBall_COLOR != None :
        color_str = None
        if FirstMovingBall_COLOR == BALL_COLOR['RED'] :
            color_str = 'RED'
        elif FirstMovingBall_COLOR == BALL_COLOR['WHITE'] :
            color_str = 'WHITE'
        elif FirstMovingBall_COLOR == BALL_COLOR['YELLOW'] :
            color_str = 'YELLOW'
        print_color_str = "YourBall : " + color_str

        cv2.putText(image, print_color_str, (900, 70), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2)
        if Waiting_start_time != None : 
            print_waitingtime_str = "Waiting Time : " + str((now - Waiting_start_time).total_seconds())
            cv2.putText(image, print_waitingtime_str, (900, 110), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2)

    cv2.imshow(window_name, image[:, :, [2, 1, 0]]) # RGB -> BGR변환. opencv는 BGR을 사용한다

    return ori_image[:, :, [2, 1, 0]], score, FirstMovingBall_COLOR, Waiting_start_time, ball_crush_stack