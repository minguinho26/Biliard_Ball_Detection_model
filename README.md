# Biliard_Ball_Detection

YOLOv3을 이용한 당구공 탐지 프로그램입니다. PyTorch를 기반으로 짜여진 YOLOv3를 직접 제작한 데이터셋으로 학습시켜 당구공을 object detection하는 모델을 만들었습니다. <br> <br>

## 알고리즘 
객체탐지을 목적으로 만든 딥러닝 네트워크 중 대표적인 모델인 [YOLOv3](https://arxiv.org/pdf/1804.02767.pdf)으로 **공의 대략적인 위치**를 둘러싼 bounding box(줄여서 bbox)를 추출합니다. 

<img width="363" alt="스크린샷 2022-02-21 오후 9 19 03" src="https://user-images.githubusercontent.com/50979281/154954133-8f9a13f2-1c83-4cc2-b7ff-f0804dce474e.png">

이렇게 추출한 영역에서 opencv의 [cv2.HoughCircles()](https://docs.opencv.org/3.4/da/d53/tutorial_py_houghcircles.html)를 사용해 공의 정확한 위치를 알아냅니다. 이 때 하나의 bbox내에 여러개의 공이 들어있는 경우가 있기 때문에 HoughCircles()을 사용하기 전에 RGB 채널을 이용해 특정 색상(빨간색, 흰색, 노란색)을 검출하는 연산을 수행하여 원하는 공을 쉽게 검출할 수 있게 만들어줍니다. 

> 위 사진에서 'yellow_bbox_window_ori'에 있는 이미지가 특정 색상(노란색)을 검출하기 전이고 'yellow_bbox_window'에 있는 이미지가 특정 색상만 검출한 이미지입니다. 

테스트 결과는 아래 '4. 테스트 결과'에서 확인하실 수 있습니다. 

## YOLOv3 구현코드 출처
YOLOv3의 backbone 네트워크인 DarkNet을 학습시키는데 필요한 데이터셋의 크기와 학습 시간, 내가 원하는 성능만큼 학습이 제대로 안될 수 있는 점을 고려해 깃허브에 올라와있는 YOLOv3 구현코드를 사용했습니다. 사전학습된 DarkNet을 기반으로 설계된 YOLOv3를 따로 제작한 '당구공 데이터셋'으로 재학습 시켰고 학습에 관한 내용은 아래 '2. 학습'에서 확인하실 수 있습니다. 

YOLOv3 구현코드 출처 : https://github.com/eriklindernoren/PyTorch-YOLOv3

<br>

## 레포지토리 구성(2022.2.6기준)

<br>

|파일명|설명|
|---|---|
|README.md|현재 읽고 계시는 파일입니다. 레포지토리에 대한 설명을 담고 있습니다.|
|update_history.md|프로그램의 개선 과정에서 기록할만한 것들을 따로 기록해놓은 파일입니다.|
|ball_detect_program.py|메인 파일입니다. 해당 파이썬 파일을 실행하여 당구공 탐지 프로그램을 사용하실 수 있습니다.|
|codes|프로그램 실행에 필요한 코드들을 모아놓은 폴더입니다.|
|yolov3|객체 탐지에 사용하는 YOLOv3의 모델을 구성하는데 필요한 파일과 학습시킨 parameter들을 모아놓은 파일을 저장하는 곳입니다.|
|util_codes.py|프로그램 작성 중에 유용했던 메소드들을 정리한 파이썬 파일입니다.|


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

### 개발, 실행 환경 구성

<img height="800" alt="스크린샷 2022-01-30 오전 1 02 44" src="https://user-images.githubusercontent.com/50979281/151799427-2e31be67-1df3-43b1-88aa-8b687fce1788.JPG">


위와 같이 당구대 위에 있는 천장을 뜯은 뒤 카메라를 설치하였습니다. 

<br>

## 1. 데이터셋 : 자체 제작 데이터셋
당구대를 천장에서 촬영 후 라벨링 작업을 수행하여 제작했습니다. 라벨링에 사용한 프로그램은 [labelme](https://github.com/wkentaro/labelme)입니다.



라벨 데이터 1             |  라벨 데이터 2
:-------------------------:|:-------------------------:
![](https://user-images.githubusercontent.com/50979281/151668240-07632f54-ee64-4a10-b47c-dc769a622d8e.png) | ![](https://user-images.githubusercontent.com/50979281/151668281-ea971d14-925e-4beb-b1d0-a6bc8941cd4d.png)


<br>

labelme를 이용해 모든 이미지에 대한 라벨링을 수행한 뒤, YOLOv3가 요구하는 [416 x 416]의 이미지 크기에 맞춰 이미지와 라벨 데이터를 변경했습니다. 
<br>
레포지토리에 있는 util_codes.py의 convert_labelmeAugFile_to_YOLOv3_AugFile()가 변경하는 일을 수행합니다.

<br>

**[발전 과정]** 

데이터셋은 3번의 개선과정을 거쳐 4번째 데이터셋을 최종 데이터셋으로 사용하였습니다. 

1. [빨간공, 흰공, 노란공] -> 손이랑 큐대를 공으로 인식하는 경우가 많이 발생
2. [손, 큐대, 빨간공, 흰공, 노란공] -> 손이랑 큐대를 공으로 인식하지 않으나 하나로 모인 공들을 제대로 인식하지 못함
3. [손, 큐대, (공 2개가 모인 것), (공 3개가 모인 것), 빨간공, 흰공, 노란공] -> 이전 데이터셋보다 공 탐지 성능이 증가했으나 하나로 모인 공을 제대로 인식하지 못하는 문제는 완전히 고쳐지지 않음. 새로운 데이터셋을 만들어서 다시 학습시킬 계획(22.01.31)
4. [손, 큐대, (공 2개가 모인 것), 빨간공, 흰공, 노란공, 움직이는 빨간공, 움직이는 흰공, 움직이는 노란공] -> 현재까지 가장 높은 성능을 가진 모델을 학습시키는 데이터셋. 공이 곂치면 하나의 공만 인식되는 문제와 빠르게 움직이는 공을 인식하지 못하는 문제가 남아있지만 이전에 비하면 상당히 개선되었음. 데이터셋 크기와 장비의 한계로 판단되며 현재 데이터셋 크기를 2,896개에서 3,550개로 늘린 뒤 학습을 진행중.(22.02.07)

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

검증 데이터셋을 이용해 측정한 모델의 성능은 다음과 같습니다. (mAP = 0.68002, 2021.02.17 기준)

| Type        | Value                |
|-------------|----------------------|
| IoU loss    | 0.13242006301879883  |
| Object loss | 0.030031338334083557 |
| Class loss  | 0.3855709731578827   |
| Batch loss  | 0.5480223894119263   |


| Index | Class              | AP      |
|-------|--------------------|---------|
| 0     | biliard_stick      | 0.36440 |
| 1     | hand               | 0.71033 |
| 2     | two_balls          | 0.47845 |
| 3     | three_balls        | 0.55114 |
| 4     | red_ball           | 0.97533 |
| 5     | white_ball         | 0.97828 |
| 6     | yellow_ball        | 0.97254 |
| 7     | moving_red_ball    | 0.56465 |
| 8     | moving_white_ball  | 0.52504 |
| 9     | moving_yellow_ball | 0.65193 |


| Type                 | Value    |
|----------------------|----------|
| validation precision | 0.582968 |
| validation recall    | 0.790646 |
| validation mAP       | 0.680018 |
| validation f1        | 0.661801 |



<br>

## 3. 실행방법

**1. 해당 레포지토리를 다운로드 받습니다. (Code -> Download ZIP)**
   <img width="800" alt="스크린샷 2022-02-06 오전 11 47 35" src="https://user-images.githubusercontent.com/50979281/152665947-2b44fa19-8212-45c7-b36f-bb1d08632ece.png">

<br><br>
**2. 다운로드 받은 zip파일을 압축해제합니다.** 
   <img width="800" alt="스크린샷 2022-02-06 오전 11 53 51" src="https://user-images.githubusercontent.com/50979281/152666050-a14a7d70-834d-4779-8315-f4f06d5e2159.png">

<br><br>
**3. [가중치 파일](https://drive.google.com/file/d/1hSL4T-Lo9RUfxuKP3n9zCtKsuUwDJ83W/view?usp=sharing)을 다운받은 뒤 yolov3 폴더에 넣어줍니다.**
   <img width="800" alt="스크린샷 2022-02-06 오전 11 52 07" src="https://user-images.githubusercontent.com/50979281/152665998-e5f8593f-d118-476e-95e4-8ce06816378f.png">

<br><br>
**4. 터미널을 실행 후 cd 명령어를 이용해 레포지토리가 있는 경로로 이동 후 'python ball_detect_program.py'를 입력해 프로그램을 실행합니다.** 
   <img width="800" alt="스크린샷 2022-02-06 오후 12 21 28" src="https://user-images.githubusercontent.com/50979281/152666594-0d4ca5d7-9951-442d-b63d-21ed1658e042.png">
**5. 실행 후 키보드 입력을 영어로 전환 후 q를 두 번 누르면 프로그램이 종료됩니다. (두 번 눌렀을 때 종료가 안된다면 여러번 눌러보시길 바랍니다.)** <br><br>


>Note : 프로그램을 실행하려면 PyTorch와 CUDA가 필요하며 **PyTorch 1.9.0과 CUDA 11.2**를 사용하시는걸 권장합니다. <br> 설치방법 참고 : https://medium.com/analytics-vidhya/install-cuda-11-2-cudnn-8-1-0-and-python-3-9-on-rtx3090-for-deep-learning-fcf96c95f7a1


<br>

## 4. 테스트 결과
학습시킨 YOLOv3으로 당구공 탐지를 수행했을 때 결과는 다음과 같습니다. (22.02.21)

<img width="1545" alt="스크린샷 2022-02-21 오후 6 19 06" src="https://user-images.githubusercontent.com/50979281/154955799-310be188-5fce-4698-a429-3f53fd5348ce.png">


YOLOv3가 공의 특징(특성 색갈의 픽셀이 여러개 모여있는 패턴)을 잘 탐지하는 점과 HoughCircles가 '구형'을 탐지한다는 점을 조합했을 때 성능, 속도 두가지 면에서 긍정적인 효과가 나올 것이라 판단하였는데 생각보다 결과가 잘나오고 있습니다. 

<br> <br>

## 5. 개선해야될 부분
두 공이 붙어있을 경우에 대해 학습했으나 두 공이 붙어있어도 한 공으로 인식하는 경우가 계속 발생했습니다. <br>

두 공이 곂쳤는데 하나의 공만 인식됨             |  두 공이 하나의 곧으로 인식됨
:-------------------------:|:-------------------------:
<img width="600" alt="스크린샷 2022-01-30 오전 1 02 44" src="https://user-images.githubusercontent.com/50979281/153020343-13e598d5-155d-4506-8734-1ed505dbed2c.jpg"> |  <img width="600" alt="스크린샷 2022-01-30 오전 1 02 44" src="https://user-images.githubusercontent.com/50979281/153021028-bc8b515d-1a4c-456a-ac92-46eeaeff1fe7.jpg">

<br>

 이를 해결하기 위해 데이터셋의 라벨링 작업을 다시 하거나 데이터셋의 크기를 키우는 작업 등을 수행하였습니다. 그리고 공 검출방식에 대한 변화도 수행하였습니다. 
 
  개선 작업에 관한 자세한 내용은 [update_history.md](./update_history.md)에서 확인하실 수 있습니다.
