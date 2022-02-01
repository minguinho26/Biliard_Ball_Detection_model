# Biliard_Ball_Detection

YOLOv3을 이용한 당구공 탐지 프로그램입니다. PyTorch를 기반으로 짜여진 YOLOv3를 직접 제작한 데이터셋으로 학습시켜 당구공을 object detection하는 모델을 만들었습니다. <br> <br>

데이터셋, 테스트 코드 제작자 : 김민규(minkyu4506@gmail.com)

YOLOv3 구현코드 출처 : https://github.com/eriklindernoren/PyTorch-YOLOv3

<br>

## 0. 개발환경
<br>

### [1] 컴퓨터 
**OS** : Ubuntu 20.04.3 LTS <br>
**GPU** : RTX 3060 (vram : 12GB) <br>
**PyTorch** : 1.9.0 <br>
**CUDA** : 11.2 <br>

### [2] 그 외 장치
**Camera** : Microsoft® LifeCam Studio(TM)

<br>

### 개발환경 구성

![IMG_0280](https://user-images.githubusercontent.com/50979281/151799427-2e31be67-1df3-43b1-88aa-8b687fce1788.JPG)

위와 같이 당구대 위에 있는 천장을 뜯은 뒤 카메라를 설치하였습니다. 

<br>

## 1. 데이터셋 : 자체 제작 데이터셋
당구대를 천장에서 촬영 후 라벨링 작업을 수행하여 제작했습니다. 라벨링에 사용한 프로그램은 [labelme](https://github.com/wkentaro/labelme)입니다.

<img width="1375" alt="스크린샷 2022-01-30 오전 1 02 44" src="https://user-images.githubusercontent.com/50979281/151668240-07632f54-ee64-4a10-b47c-dc769a622d8e.png">

<br>

<img width="1375" alt="스크린샷 2022-01-30 오전 1 11 53" src="https://user-images.githubusercontent.com/50979281/151668281-ea971d14-925e-4beb-b1d0-a6bc8941cd4d.png">

<br>

labelme를 이용해 모든 이미지에 대한 라벨링을 수행한 뒤, [416 x 416] 크기에 맞춰 이미지와 라벨 데이터를 변경하였습니다. 

<br>

**[발전 과정]**

데이터셋은 2번의 개선과정을 거쳐 3번째 데이터셋을 최종 데이터셋으로 사용하였습니다. 

1. [빨간공, 흰공, 노란공] -> 손이랑 큐대를 공으로 인식하는 경우가 많이 발생
2. [손, 큐대, 빨간공, 흰공, 노란공] -> 손이랑 큐대를 공으로 인식하지 않으나 하나로 모인 공들을 제대로 인식하지 못함
3. [손, 큐대, (공 2개가 모인 것), (공 3개가 모인 것), 빨간공, 흰공, 노란공] -> 이전 데이터셋보다 공 탐지 성능이 증가했으나 하나로 모인 공을 제대로 인식하지 못하는 문제는 완전히 고쳐지지 않음. 새로운 데이터셋을 만들어서 다시 학습시킬 계획(22.01.31)

## 2. 학습

학습에 사용한 Hyperparameter들은 다음과 같습니다. 

>**epochs** = 300 <br>
**batch size** = 16 <br>
**momentum** = 0.9 <br>
**weight decay** = 0.0005 <br> 
**burn_in** = 1000 # 학습률 관련 Hyperparameter

여기서 burn_in은 학습률 조정에 사용하는 값입니다. 미니 배치 단위로 학습을 수행할 때마다 학습률을 (지금까지 학습에 사용했던 미니 배치의 개수/ burn_in )으로 수정했습니다. 

즉, 학습을 할 때마다 학습률을 조금씩 상승시킨 것입니다.

<br>

### 학습 후 성능
<br>

검증 데이터셋을 이용해 측정한 모델의 성능은 다음과 같습니다. (mAP = 0.6148)

|객체 종류|AP|
|------ |-------|
|큐대|0.05052|
|손|0.55097|
|공 2개 모인 것|0.71080|
|공 3개 모인 것|0.04444|
|**빨간공**|0.97318|
|**흰공**|0.97668|
|**노란공**|0.99698|

제가 학습시킨 YOLOv3의 주목적은 '공을 잘 탐지하는 것'입니다. <br>
빨간공, 흰공, 노란공 외의 다른 객체들은 '공이 아닌 것을 공으로 잘못 판정하는 일'을 최대한 줄이기 위해 탐지 대상으로 추가한 것들이며 실제로 탐지 대상을 늘릴 수록 공 탐지에 robust한 모습이 강해지는 것을 확인할 수 있었습니다.


<br>

## 3. 실행방법

**1. 해당 레포지토리를 다운로드 받습니다. (Code -> Download ZIP)**
   <img width="903" alt="다운로드 방법" src="https://user-images.githubusercontent.com/50979281/151987129-d42eef73-1840-49f2-9989-53fff4cb489a.png">
**2. 다운로드 받은 zip파일을 압축해제합니다.** <br><br>
**3. [다운로드 링크](https://drive.google.com/file/d/1e7ddvkeBNNk3MQPlJ10klacODzXNdRC6/view?usp=sharing)에서 가중치 파일을 다운받은 뒤 yolov3 폴더에 넣어줍니다.**
   <img width="861" alt="가중치 파일 저장" src="https://user-images.githubusercontent.com/50979281/151987811-b98042cc-c1b8-4741-b373-7595d2f43a9e.png">
**4. yolov3의 상위 폴더에 있는 Ball_detect_program를 실행합니다.**


**5. 실행 후 키보드 입력을 영어로 전환 후 q를 두 번 누르면 프로그램이 종료됩니다. (두 번 눌렀을 때 종료가 안된다면 여러번 눌러보시길 바랍니다.)** <br><br>


>Note : 프로그램을 실행하려면 PyTorch와 CUDA가 필요하며 **PyTorch 1.9.0과 CUDA 11.2**를 사용하시는걸 권장합니다. <br> 설치방법 참고 : https://medium.com/analytics-vidhya/install-cuda-11-2-cudnn-8-1-0-and-python-3-9-on-rtx3090-for-deep-learning-fcf96c95f7a1


<br>

## 4. 테스트 결과
학습시킨 YOLOv3으로 당구공 탐지를 수행했을 때 결과는 다음과 같습니다. 

 ![테스트 화면 1](https://user-images.githubusercontent.com/50979281/151748223-96fdd8ff-80c2-4035-bfcd-23a3f1970d26.jpg)

<br>

![테스트 화면 2](https://user-images.githubusercontent.com/50979281/151748239-414f3867-c2fb-4017-a956-02cd6e9b7c94.jpg)

당구대에 흰공이 없거나 흰공의 confidence가 매우 낮아 nms를 통과하지 못하는 경우가 아니면 손과 당구대를 공으로 인식하는 일이 없었습니다. 

<br> <br>

## 4. 개선해야될 부분
두 공이 붙어있을 경우에 대해 학습했으나 두 공이 붙어있어도 한 공으로 인식하는 경우가 계속 발생했습니다. <br>


![아쉬운 장면 1](https://user-images.githubusercontent.com/50979281/151749899-c323de34-7a0a-4f8c-91cb-24b0f7853ca8.jpg)
![아쉬운 장면 2](https://user-images.githubusercontent.com/50979281/151749909-fd4fec2c-4566-44c7-8045-ca232aad3fcd.jpg)


이를 해결하기 위해 데이터셋의 라벨링 작업을 다시 하거나 데이터셋의 크기를 키우는 방안을 생각중입니다. 




