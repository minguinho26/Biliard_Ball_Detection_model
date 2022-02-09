# update_history.md
프로그램의 개선 과정에서 기록할만한 것들을 기록해놓은 .md 파일입니다.

**22.01.21** : [빨간공, 노란공, 흰공]만 라벨링한 1100개의 데이터로 구성된 데이터셋으로 학습시켰을 때 결과입니다.


| Type        | Value                 |
|-------------|-----------------------|
| IoU loss    | 0.019257018342614174  |
| Object loss | 0.0049205804243683815 |
| Class loss  | 0.0026953278575092554 |
| Batch loss  | 0.026872927322983742  |


| Index | Class       | AP      |
|-------|-------------|---------|
| 0     | white_ball  | 0.95239 |
| 1     | yellow_ball | 0.98411 |
| 2     | red_ball    | 0.99498 |

---- mAP 0.97716 ----


| Type                 | Value    |
|----------------------|----------|
| validation precision | 0.898061 |
| validation recall    | 0.987879 |
| validation mAP       | 0.977156 |
| validation f1        | 0.939965 |

성능은 아주 잘나왔으나 실제 테스트 결과, 손이나 큐대를 공으로 판정하는 경우가 아주 많이 발생하여 라벨링 대상에 손과 큐대를 추가하였고 데이터셋의 크기도 1100장에서 2896장으로 늘렸습니다. 

<br>

**22.01.29** : 크기를 늘린 데이터셋을 가지고 학습을 수행했을 때 성능입니다. 

| Index | Class              | AP      |
|-------|--------------------|---------|
| 0     | biliard_stick      | 0.00000 |
| 1     | hand               | 0.39229 |
| 2     | red_ball           | 0.97814 |
| 3     | white_ball         | 1.04076 |
| 4     | yellow_ball        | 0.96017 |

---- mAP 0.67427 ----


| Type                 | Value    |
|----------------------|----------|
| validation precision | 0.598959 |
| validation recall    | 0.712812 |
| validation mAP       | 0.674272 |
| validation f1        | 0.643860 |

이번에는 여러개의 공이 모였을 때 하나의 공만 탐지하는 현상을 발견해 'two_balls', 'three_balls'를 추가로 라벨링 하였습니다. 

<br>

**22.01.30** : 'two_balls', 'three_balls'를 추가하고 학습을 시켰으며 측정된 성능은 다음과 같습니다. 

|Index|Class|AP|
|----|-----|-----|
| 0     | biliard_stick      | 0.05052 |
| 1     | hand               | 0.55097 |
| 2     | two_balls          | 0.71080 |
| 3     | three_balls        | 0.04444 |
| 4     | red_ball           | 0.97318 |
| 5     | white_ball         | 0.97668 |
| 6     | yellow_ball        | 0.99698 |

---- mAP 0.6148 ----


| Type                 | Value    |
|----------------------|----------|
| validation precision | 0.575732 |
| validation recall    | 0.641533 |
| validation mAP       | 0.599258 |
| validation f1        | 0.594709 |


<br><br>
**22.02.03** : 공을 라벨링한 bbox의 크기가 공 전체가 아닌 내부만 포함한 경우가 많아 '공의 경계'를 제대로 학습하지 못했다는 사실을 확인했습니다. 그래서 bbox의 크기를 늘렸습니다. 그리고 빠르게 움직이는 공에 대해 moving_ball이라고 따로 라벨링을 하였으며 epoch 71에서 측정된 성능은 다음과 같습니다. 


|Index|Class|AP|
|----|-----|-----|
| 0     | biliard_stick             | 0.00000 |
| 1     | hand                      | 0.60066 |
| 2     | two_balls                 | 0.04264 |
| 3     | three_balls               | 0.00000 |
| 4     | red_ball                  | 0.92029 |
| 5     | white_ball                | 0.92569 |
| 6     | yellow_ball               | 0.92270 |
| 7     | moving_red_ball           | 0.55229 |
| 8     | moving_white_ball         | 0.44594 |
| 9     | moving_yellow_ball        | 0.53204 |

<br><br>

**22.02.04** : 공들의 bbox 크기를 일괄적으로 상하좌우 1.5씩 늘리고 학습을 수행한 결과, 노란공이 큐대와 가까운 위치에 있어야 공으로 인식이 된다는 사실을 발견했습니다. 그래서 bbox를 상하좌우 1만큼만 늘리고 공을 표시하는 코드를 수정하기로 했습니다. 2022.02.04 오전 10시 50분부터 수정한 데이터셋으로 학습을 다시 시작했습니다. 

<br><br>

**22.02.05** : 공들의 bbox의 크기를 상하좌우로 1픽셀씩만 늘려도 같은 현상이 발생하여 bbox늘리는 것을 하지 않았습니다. 대신 two_balls와 three_balls의 기준이 일관적이지 못하다는걸 발견했고 특히 three_balls는 라벨링을 해도 탐지를 하는 경우가 없었기 때문에 라벨링 대상에 three_balls를 제외, two_balls이라 판정하는 기준을 보다 엄격하게 정하여 데이터셋에 적용, 학습을 진행했습니다. 

학습 결과는 다음과 같습니다.


|Type|Value|
|----|-----|
|IoU loss | 0.01940111815929413  |
|Object loss | 0.006405044347047806  |
|Class loss | 0.0534568727016449  |
|Batch loss | 0.07926303148269653  |


|Index|Class|AP|
|----|-----|-----|
| 0     | biliard_stick      | 0.05381 |
| 1     | hand               | 0.40233 |
| 2     | two_balls          | 0.51695 |
| 3     | red_ball           | 0.81567 |
| 4     | white_ball         | 0.58405 |
| 5     | yellow_ball        | 0.95774 |
| 6     | moving_red_ball    | 0.47643 |
| 7     | moving_white_ball  | 0.26847 |
| 8     | moving_yellow_ball | 0.28966 |
| 9     | None               | 0.00000 |


---- mAP 0.43651 ----


| Type                 | Value    |
|----------------------|----------|
| validation precision | 0.367735 |
| validation recall    | 0.634910 |
| validation mAP       | 0.436511 |
| validation f1        | 0.425193 |

<br>

**22.02.06** : 빨간공, 노란공은 잘 탐지되나 흰공이 제대로 인식 안되는 현상을 발견했습니다. <br> 이를 해결하기 위해 흰공의 bbox만 상하좌우로 1픽셀씩 늘려 데이터셋을 재구성 하였습니다. 학습결과는 다음과 같습니다.

| Type        | Value                |
|-------------|----------------------|
| IoU loss    | 0.05444442108273506  |
| Object loss | 0.010509137064218521 |
| Class loss  | 0.0764257088303566   |
| Batch loss  | 0.14137926697731018  |


| Index | Class              | AP      |
|-------|--------------------|---------|
| 0     | biliard_stick      | 0.01337 |
| 1     | hand               | 0.53345 |
| 2     | two_balls          | 0.57100 |
| 3     | red_ball           | 0.97780 |
| 4     | white_ball         | 0.98380 |
| 5     | yellow_ball        | 0.97419 |
| 6     | moving_red_ball    | 0.51518 |
| 7     | moving_white_ball  | 0.41327 |
| 8     | moving_yellow_ball | 0.49014 |
| 9     | None               | 0.00000 |

---- mAP 0.54722 ----

| Type                 | Value    |
|----------------------|----------|
| validation precision | 0.465478 |
| validation recall    | 0.697178 |
| validation mAP       | 0.547221 |
| validation f1        | 0.508614 |

테스트 결과, 흰공 탐지는 원활이 잘되나 노란공이 당구공의 오른쪽에 있을 때 탐지가 잘 안되는 현상을 발견했습니다. 그래서 이를 개선하기 위한 작업을 수행중입니다.

<br>

**22.02.07** : 데이터셋의 크기를 2,896개에서 3,550개로 늘렸습니다. 현재 학습중이며 Epoch 50에서 측정된 성능은 다음과 같습니다.

| Type        | Value                |
|-------------|----------------------|
| IoU loss    | 0.056652989238500595 |
| Object loss | 0.012153341434895992 |
| Class loss  | 0.10598105192184448  |
| Batch loss  | 0.17478737235069275  |

| Index | Class              | AP      |
|-------|--------------------|---------|
| 0     | biliard_stick      | 0.08430 |
| 1     | hand               | 0.32893 |
| 2     | two_balls          | 0.47810 |
| 3     | red_ball           | 0.96002 |
| 4     | white_ball         | 1.01945 |
| 5     | yellow_ball        | 0.90125 |
| 6     | moving_red_ball    | 0.38589 |
| 7     | moving_white_ball  | 0.46913 |
| 8     | moving_yellow_ball | 0.15293 |

---- mAP 0.53111 ----

| Type                 | Value    |
|----------------------|----------|
| validation precision | 0.567310 |
| validation recall    | 0.640428 |
| validation mAP       | 0.531111 |
| validation f1        | 0.557080 |

Note : mAP가 비슷한 것은 None의 AP를 계산 대상에서 제외했기 때문입니다. 객체별 정확도를 비교하시면 됩니다. 

학습 후 측정된 성능 


| Type        | Value                |
|-------------|----------------------|
| IoU loss    | 0.051199983805418015 |
| Object loss | 0.021511608734726906 |
| Class loss  | 0.09351477771997452  |
| Batch loss  | 0.1662263721227646   |



| Index | Class              | AP      |
|-------|--------------------|---------|
| 0     | biliard_stick      | 0.03810 |
| 1     | hand               | 0.43476 |
| 2     | two_balls          | 0.45751 |
| 3     | red_ball           | 0.96403 |
| 4     | white_ball         | 1.00225 |
| 5     | yellow_ball        | 0.95251 |
| 6     | moving_red_ball    | 0.45313 |
| 7     | moving_white_ball  | 0.55332 |
| 8     | moving_yellow_ball | 0.44562 |

---- mAP 0.58903 ----

| Type                 | Value    |
|----------------------|----------|
| validation precision | 0.530989 |
| validation recall    | 0.712010 |
| validation mAP       | 0.589028 |
| validation f1        | 0.596022 |
|----------------------|----------|

**22.02.08**  : class에 관한 loss를 계산할 때 two_balls, three_balls가 있는 이미지를 가지고 계산한 loss를 키웠습니다. 현재 사용중인 YOLOv3은 배치 단위로 target을 받고 YOLOv3가 3종류의 scale에서 출력한 bbox들을 가지고 Loss를 계산하는데 해당 bbox들이 two_balls, three_balls를 예측한 경우가 많아질 수록 해당 배치의 loss크기가 커지는 것입니다. 현재 2가지 방식을 생각했으며 코드로 설명하면 다음과 같습니다.

~~~python
# Hot one class encoding
t = torch.zeros_like(ps[:, 5:], device=device) # targets 
t[range(num_targets), tcls[layer_index]] = 1
                

# 방식 1 : 배치 단위로 받을 때 배치에 two_balls, three_balls가 들어있는 데이터 비율을 loss계산에 반영
multi_ball_ratio = t[t[:,2] == 1].size()[0] / t.size()[0]
lcls += BCEcls(ps[:, 5:], t) + 0.01 * multi_ball_ratio * lcls += BCEcls(ps[:, 5:], t) 


# 방식 2 : two_balls, three_balls가 있는 데이터에 관한 loss를 구한 뒤 loss계산에 추가로 반영
idx_multiballs = t[:,2].nonzero(as_tuple=True)[0]
t_multi_balls = t[idx_multiballs,:]
ps_multi_balls = ps[idx_multiballs,5:]
                
              
# Use the tensor to calculate the BCE loss
lcls += BCEcls(ps[:, 5:], t) # 기존의 loss
if ps_multi_balls.size()[0] > 0 and t_multi_balls.size()[0] : # two_balls, three_balls가 있는 데이터가 있으면 loss에 더함
    lcls += 0.1 * BCEcls(ps_multi_balls, t_multi_balls)
~~~

그리고 데이터 라벨링을 추가로 수행하여 총 3612개의 데이터가 들어있는 데이터셋이 되었으며 공 3개가 모이는 경우가 상닫히 많아 'three_balls'도 탐지 대상으로 다시 추가 후 학습을 진행하였습니다. 측정된 성능은 다음과 같습니다. 

<br>

1. 방식 1로 구한 loss로 학습시킬 경우

    | Type        | Value                |
    |-------------|----------------------|
    | IoU loss    | 0.05494890734553337  |
    | Object loss | 0.013550731353461742 |
    | Class loss  | 0.14103847742080688  |
    | Batch loss  | 0.20953811705112457  |


    | Index | Class             | AP      |
    |-------|-------------------|---------|
    | 0     | biliard_stick     | 0.00518 |
    | 1     | hand              | 0.53844 |
    | 2     | two_balls         | 0.45526 |
    | 3     | three_balls       | 0.00306 |
    | 4     | red_ball          | 0.99263 |
    | 5     | white_ball        | 1.03725 |
    | 6     | yellow_ball       | 0.86106 |
    | 7     | moving_red_ball   | 0.49443 |
    | 8     | moving_white_ball | 0.39112 |

    ---- mAP 0.53094 ----

    | Type                 | Value    |
    |----------------------|----------|
    | validation precision | 0.501264 |
    | validation recall    | 0.617272 |
    | validation mAP       | 0.530936 |
    | validation f1        | 0.541564 |

<br>

2. 방식 2로 구한 loss로 학습시켰을 경우(학습 중)