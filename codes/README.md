# codes
프로그램을 실행하는데 필요한 파이썬 파일들을 모아놓은 폴더입니다.

<br>

## 구성요소

<br>

### 1. YOLOv3_models.py 
 YOLOv3 구현 코드의 작성자인 [Erik Linder-Norén](https://github.com/eriklindernoren)이 작성한 YOLOv3의 구현 코드가 저장된 파일입니다. <br> YOLOv3의 클래스와(파일에서는 DarkNet이라는 클래스로 저장되어 있습니다) 클래스의 객체를 초기화하는 메소드가 저장되어 있습니다.

<br>

### 2. YOLOv3_utils.py
YOLOv3을 이용해 객체 검출을 할 때 필요한 메소드들이 들어있는 파일입니다. change_bbox_to_use()를 제외한 모든 메소드들은 [Erik Linder-Norén](https://github.com/eriklindernoren)이 작성한 메소드들이며 change_bbox_to_use()도 Erik Linder-Norén이 작성한 non_max_suppression()를 일부 수정한 메소드입니다.

<br>

### 3. detect_BiliardBall_usingYOLOv3.py
당구공 검출에 필요한 메소드들이 저장된 파일입니다. <br>
ball_detect_program.py에 들어있는 메인 함수에서 해당 파일에 있는 detect_biliard_ball()을 호출하여 촬영중인 프레임에 있는 공을 탐지한 다음 프레임에 bbox를 그리고 화면에 출력합니다.


