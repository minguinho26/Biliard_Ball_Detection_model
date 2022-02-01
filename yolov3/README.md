# YOLOv3

YOLOv3를 사용하는데 필요한 파일들이 모여있는 폴더입니다.

<br>

## 구성요소

## 1. yolov3-for-Biliard-7Classes.cfg
[큐대, 손, (공 2개 모인 것), (공 3개 모인 것), 빨간공, 흰공, 노란공]을 구분하는 YOLOv3의 구조가 저장된 파일입니다. 
<br>
파일의 구조는 다음과 같습니다.


Training            |  Layers
:-------------------------:|:-------------------------:
!<img width="300" alt="스크린샷 2022-01-31 오전 10 11 58" src="https://user-images.githubusercontent.com/50979281/151726358-e6770e86-8623-4c0b-ba0a-f3f10ebb3ddf.png">  |  <img width="172" alt="스크린샷 2022-01-31 오전 10 12 16" src="https://user-images.githubusercontent.com/50979281/151726366-b96ae73a-f6ef-4490-95a1-4bfd62c07d66.png">

Training은 학습에 필요한 Hyperparameter들의 값이고 Layers는 YOLOv3를 구성하는 레이어들의 정보가 적혀있습니다. codes/YOLOv3_models.py의 parse_model_config()과 create_modules()으로 cfg를 해석해 YOLOv3을 만듭니다.

<br>

## 2. yolov3_weights_7Classes.pth
학습시킨 YOLOv3의 가중치들을 저장한 파일입니다. 다음과 같이 load_state_dict()를 이용해 학습된 가중치를 YOLOv3에다 불러올 수 있습니다.

~~~python
model.load_state_dict(torch.load('yolov3_weights_7Classes.pth', map_location='cuda'))
~~~

<br>

만약 파일이 없으면 [링크](https://drive.google.com/file/d/1e7ddvkeBNNk3MQPlJ10klacODzXNdRC6/view?usp=sharing)에서 파일을 다운로드 받은 후 본 README가 있는 yolov3 폴더에 놔두시면 됩니다.



