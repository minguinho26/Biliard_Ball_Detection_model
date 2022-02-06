# 촬영중인 카메라에서 얻은 프레임에 있는 당구공 위치를 YOLOv3으로 탐지하는 프로그램

# 생성일 : 2022.1.21
# 제작자 : 김민규

from codes.YOLOv3_models import *
from codes.detect_BiliardBall_usingYOLOv3 import *
import numpy as np
import cv2

# YOLOv3을 불러오기 위한 메소드
def ready_for_detect() :
    # 모델의 구조가 담긴 cfg와 학습시킨 가중치들이 담긴 pth의 경로
    model_path = os.getcwd() + '/yolov3/yolov3-for-Biliard-9Classes.cfg'
    pretrained_weights = os.getcwd() + '/yolov3/yolov3_weights_9Classes.pth'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 파이토치를 이용해 만든 객체, 변수들을 gpu에서 돌릴지 cpu에서 돌릴지 결정.
    model = load_model(model_path, pretrained_weights) # 학습시킨 가중치가 저장된 YOLOv3을 불러옴
    model.eval() # YOLOv3을 테스트 모드로 변경
    return model, device

def main() :
    model, device = ready_for_detect() # YOLOv3과 파이토치 기반 객체, 변수를 돌릴 장치(cpu or gpu)에 대한 정보를 휙득

    capture = cv2.VideoCapture(0) # 연결된 카메라 혹은 비디오에서 프레임을 얻기 위해 필요한 객체. 0을 매개변수로 전달하면 연결된 카메라에서 프레임을 얻어온다

    # 카메라로 프레임을 캡처하지 못했으면 프로그램 종료
    if not capture.isOpened():
        print("Could not open webcam")
        exit()

    while True:
        # 찍고있는 카메라에서 캡처 상태와 찍고있는 프레임 얻기
        status, frame = capture.read() # status는 boolean이고 frame은 numpy array다.

        if not status:
            break
        elif cv2.waitKey(1) == ord('q') : # q 누르면 탈출
            break
        frame = frame[:, :, [2, 1, 0]] # # BGR -> RGB. numpy와 pytorch는 RGB를 사용한다
        
        detect_biliard_ball(model, frame.copy(), device, 'Detect_ball') # 공 탐지 후 화면에 보여줌
        
    capture.release() # 캡처 중지
    cv2.destroyAllWindows() # 화면 생성 중단

if __name__ == "__main__":
    main()
