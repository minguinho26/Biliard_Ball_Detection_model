# YOLOv3 코드(https://github.com/eriklindernoren/PyTorch-YOLOv3)에 있는 YOLOv3에게 당구공 데이터셋을 학습시킨 뒤 테스트하는 코드

# 생성일 : 2022.1.21
# 제작자 : 김민규
from __future__ import division

import numpy as np
import cv2
import timeit
import datetime
import copy
import math

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

TWO_BALLS_INDEX          = 2
THREE_BALLS_INDEX        = 3


# ====================================
# 전역변수 선언, 정의 =====================
# ====================================
def filter_in_BiliardField(detect_ball_coordi_group, threashold_area) :

    width_group = detect_ball_coordi_group[:,2] - detect_ball_coordi_group[:,0]
    height_group = detect_ball_coordi_group[:,3] - detect_ball_coordi_group[:,1]

    area_group = torch.multiply(width_group, height_group)

    detect_ball_coordi_group = detect_ball_coordi_group[area_group[:] <= threashold_area]

    return detect_ball_coordi_group

def get_distance(ball1, ball2) : 

    point1 = ball1.coordi_list[-1].get_center()
    point2 = ball2.coordi_list[-1].get_center()

    diff_x = abs(point1[0] - point2[0])
    diff_y = abs(point1[1] - point2[1])

    return np.sqrt(diff_x * diff_x + diff_y * diff_y)

def get_moving_angle_between_3_Frames(ball) :
    ball_coordi_now = ball.coordi_list[-1].get_center()
    ball_coordi_pre_1 = ball.coordi_list[-2].get_center()
    ball_coordi_pre_2 = ball.coordi_list[-3].get_center()

    vector_now = np.asarray([ball_coordi_now[0] - ball_coordi_pre_1[0], ball_coordi_now[1] - ball_coordi_pre_1[1]])
    vector_pre = np.asarray([ball_coordi_pre_1[0] - ball_coordi_pre_2[0], ball_coordi_pre_1[1] - ball_coordi_pre_2[1]])

    vectors_dot = np.dot( vector_now, vector_pre)

    moving_angle = np.arccos( vectors_dot / (np.linalg.norm(vector_now) * np.linalg.norm(vector_pre)))

    return moving_angle

def determine_crush_stack(ball_crush_stack, FirstMovingBall_COLOR, BALLS, now) :
    target_color_idx = FirstMovingBall_COLOR # crush_stack을 올릴 공의 index
    target_ball_1_idx = None
    target_ball_2_idx = None

    red_ball = BALLS[0]
    white_ball = BALLS[1]
    yellow_ball = BALLS[2]

    myball = None
    target_ball_1 = None
    target_ball_2 = None

    target_ball1_crush_count_binary = 0
    target_ball2_crush_count_binary = 0

    if target_color_idx == BALL_COLOR['RED'] :
        target_ball_1_idx = BALL_COLOR['WHITE']
        target_ball_2_idx = BALL_COLOR['YELLOW']
        myball = red_ball
        target_ball_1 = white_ball
        target_ball_2 = yellow_ball
        
    elif target_color_idx == BALL_COLOR['WHITE'] :
        target_ball_1_idx = BALL_COLOR['YELLOW']
        target_ball_2_idx = BALL_COLOR['RED']
        myball = white_ball
        target_ball_1 = yellow_ball
        target_ball_2 = red_ball
    elif target_color_idx == BALL_COLOR['YELLOW'] :
        target_ball_1_idx = BALL_COLOR['RED']
        target_ball_2_idx = BALL_COLOR['WHITE']
        myball = yellow_ball
        target_ball_1 = red_ball
        target_ball_2 = white_ball

    if sum(ball_crush_stack) == 1 :
        if type(target_ball_1.movingStartTime) == type(now) or type(target_ball_2.movingStartTime) == type(now) :
            if type(target_ball_1.movingStartTime) == type(now) and type(target_ball_2.movingStartTime) == type(now) : # 한 번에 두개의 공을 침                 
                if get_distance(myball, target_ball_1) < get_distance(myball, target_ball_2):
                    target_ball1_crush_count_binary = 1
                elif get_distance(myball, target_ball_2) < get_distance(myball, target_ball_1):
                    target_ball2_crush_count_binary = 1
                elif abs(get_distance(myball, target_ball_2) - get_distance(myball, target_ball_1)) <= 3.0 :
                    target_ball1_crush_count_binary = 1
                    target_ball2_crush_count_binary = 1
            else : 
                if get_distance(myball, target_ball_1) < get_distance(myball, target_ball_2) :    
                    target_ball1_crush_count_binary = 1
                else :
                    target_ball2_crush_count_binary = 1    
    elif sum(ball_crush_stack) == 2 :
        # think the situation when ball i didn't hit using stick hit other ball
        if type(target_ball_1.movingStartTime) == type(now) and type(target_ball_2.movingStartTime) == type(now) : # 공 근처를 지나가기만 해도 움직인 것으로 판정됨  
            if get_distance(myball, target_ball_1) < get_distance(myball, target_ball_2) :    
                target_ball1_crush_count_binary = 1
            else :
                target_ball2_crush_count_binary = 1   
    if ball_crush_stack[target_ball_1_idx] == 0 :
        ball_crush_stack[target_ball_1_idx] = target_ball1_crush_count_binary
    if ball_crush_stack[target_ball_2_idx] == 0 :
        ball_crush_stack[target_ball_2_idx] = target_ball2_crush_count_binary

    return ball_crush_stack

# 공 탐지 + 출력, 충돌여부 판단, 점수 득점여부 판단 등 모든걸 수행하는 함수
# 모델을 학습시킬 때 RGB형식의 3채널 이미지를 사용하게끔 학습시켰기 때문에 opencv에서 얻은 BGR이미지를 RGB이미지로 변경 후 YOLOv3에 넣는다
def detect_biliard_ball(model, image, device, window_name, BALLS, score, FirstMovingBall_COLOR, Waiting_start_time, ball_crush_stack,  img_size = 416) : 

    time_start = timeit.default_timer() # start time

    ori_image = image.copy() # 공을 그리기 전의 이미지

    # 공들의 좌표를 출력하기 위해 사용하는 BALL 클래스의 객체들
    red_ball = BALLS[0]
    white_ball = BALLS[1]
    yellow_ball = BALLS[2]
    
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
    threashold_area_ball = 5000.0

    if red_ball_coordi_group.size()[0] > 0 :
        red_ball_coordi_group = filter_in_BiliardField(red_ball_coordi_group, threashold_area_ball)
    if white_ball_coordi_group.size()[0] > 0 :
        white_ball_coordi_group = filter_in_BiliardField(white_ball_coordi_group, threashold_area_ball)
    if yellow_ball_coordi_group.size()[0] > 0 :
        yellow_ball_coordi_group = filter_in_BiliardField(yellow_ball_coordi_group, threashold_area_ball)
        
    if moving_red_ball_coordi_group.size()[0] > 0 :
        moving_red_ball_coordi_group = filter_in_BiliardField(moving_red_ball_coordi_group, threashold_area_ball)
    if moving_white_ball_coordi_group.size()[0] > 0 :
        moving_white_ball_coordi_group = filter_in_BiliardField(moving_white_ball_coordi_group, threashold_area_ball)
    if moving_yellow_ball_coordi_group.size()[0] > 0 :
        moving_yellow_ball_coordi_group = filter_in_BiliardField(moving_yellow_ball_coordi_group, threashold_area_ball)
   
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
        target_bbox = moving_red_ball_coordi
    else :
        target_bbox = red_ball_coordi
    isSETCOORDI_RED, FirstMovingBall_COLOR, ball_crush_stack = red_ball.set_coordi(image.copy(), target_bbox, FirstMovingBall_COLOR, ball_crush_stack)

    if moving_white_ball_coordi[4] > white_ball_coordi[4] :
        target_bbox = moving_white_ball_coordi
    else :
        target_bbox = white_ball_coordi
    isSETCOORDI_WHITE, FirstMovingBall_COLOR, ball_crush_stack = white_ball.set_coordi(image.copy(), target_bbox, FirstMovingBall_COLOR, ball_crush_stack)

    if moving_yellow_ball_coordi[4] > yellow_ball_coordi[4] :
        target_bbox = moving_yellow_ball_coordi
    else :
        target_bbox = yellow_ball_coordi
    isSETCOORDI_YELLOW, FirstMovingBall_COLOR, ball_crush_stack = yellow_ball.set_coordi(image.copy(), target_bbox, FirstMovingBall_COLOR, ball_crush_stack)

    # see_multiballs_bbox(image.copy(), [two_balls_coordi, three_balls_coordi])


    isCaptureFrame = False # 지금 사용중인 프레임을 이미지 파일로 저장할지 말지 결정하는데 쓰이는 Boolean 변수. 프레임에서 탐지가 안되는 공이 있을 경우 True가 된다.
    
    # 만약 탐지되지 않은 공이 있으면 캡처 후 데이터셋 증축에 사용
    if isSETCOORDI_RED == True and isSETCOORDI_WHITE == True and isSETCOORDI_YELLOW == True : 
        isCaptureFrame = True
    
    can_determine_crush = False
    if len(red_ball.coordi_list) > 0 and len(white_ball.coordi_list) > 0 and len(yellow_ball.coordi_list) > 0 :
        can_determine_crush = True

    now = datetime.datetime.now()
    if FirstMovingBall_COLOR != None and can_determine_crush == True : 
        ball_crush_stack = determine_crush_stack(ball_crush_stack, FirstMovingBall_COLOR, BALLS, now)

    if red_ball.status == STATUS['WAITING'] and  white_ball.status == STATUS['WAITING'] and yellow_ball.status == STATUS['WAITING'] and FirstMovingBall_COLOR != None and len(red_ball.coordi_list) > 1 and len(white_ball.coordi_list) > 1 and len(yellow_ball.coordi_list) > 1:
        if ball_crush_stack[FirstMovingBall_COLOR] == 2 : 
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
        
        if len(detect_bbox.coordi_list) > 0 :
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

        cv2.putText(image, distance_rw_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2)
        cv2.putText(image, distance_wy_str, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2)
        cv2.putText(image, distance_yr_str, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2)
        
    if FirstMovingBall_COLOR != None :
        color_str = None
        if FirstMovingBall_COLOR == BALL_COLOR['RED'] :
            color_str = 'RED'
        elif FirstMovingBall_COLOR == BALL_COLOR['WHITE'] :
            color_str = 'WHITE'
        elif FirstMovingBall_COLOR == BALL_COLOR['YELLOW'] :
            color_str = 'YELLOW'
        print_color_str = "YourBall : " + color_str

        ball_crush_stack_print = "ball_crush_stack : " + str(sum(ball_crush_stack) - 1)
        cv2.putText(image, ball_crush_stack_print, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2)

        cv2.putText(image, print_color_str, (900, 70), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2)
        if Waiting_start_time != None : 
            print_waitingtime_str = "Waiting Time : " + str((now - Waiting_start_time).total_seconds())
            cv2.putText(image, print_waitingtime_str, (900, 110), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2)

    cv2.imshow(window_name, image[:, :, [2, 1, 0]]) # RGB -> BGR변환해서 출력. opencv는 BGR을 사용한다

    return isCaptureFrame, ori_image[:, :, [2, 1, 0]], score, FirstMovingBall_COLOR, Waiting_start_time, ball_crush_stack