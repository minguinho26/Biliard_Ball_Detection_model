# YOLOv3를 구현한 코드(https://github.com/eriklindernoren/PyTorch-YOLOv3)에서 YOLOv3를 이용해 객체 검출을 할 때 필요한 메소드, 클래스들을 모아놓은 .py 파일
# 제작자 : Erik Linder-Norén(https://github.com/eriklindernoren)

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import time

import numpy as np
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

# ====================================================================
# pytorchyolo/utils/utils.py에 있던 메소드들 =============================
# ====================================================================

# YOLOv3에서 출력하는 bbox는 [x_center, y_center, width, height] 양식이다
# 허나 opencv에서 이미지에 사각형 등을 그리기 위해 필요한 위치 정보는 (x_min, y_min), (x_max, y_max) 다
# 그래서 [x_center, y_center, width, height] -> [x_min, y_min, x_max, y_max]로 변환을 해야하며 xywh2xyxy, xywh2xyxy_np는 이를 수행하는 메소드들이다.

def xywh2xyxy(x): # PyTorch의 tensor로 반환
    y = x.new(x.shape)
    # left-up
    y[..., 0] = x[..., 0] - (x[..., 2] / 2)
    y[..., 1] = x[..., 1] - (x[..., 3] / 2)
    # right_down
    y[..., 2] = x[..., 0] + (x[..., 2] / 2)
    y[..., 3] = x[..., 1] + (x[..., 3] / 2)
    return y

def xywh2xyxy_np(x): # numpy array로 반환
    y = np.zeros_like(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

# tensor를 cpu에 할당하기 위한 메소드
def to_cpu(tensor):
    return tensor.detach().cpu()

# YOLOv3에 넣은 이미지는 여러가지 전처리를 수행하기 때문에 우리가 아는 이미지가 아니다
# 즉, YOLOv3에서 얻은 bbox 정보는 전처리된 이미지에서 구한 정보이며 이를 우리가 이해할 수 있게 원본 이미지에 적합한 bbox로 변환할 필요가 있다
# rescale_boxes는 그러한 일을 수행하는 메소드다
def rescale_boxes(boxes, current_dim, original_shape): # current_dim은 YOLOv3을 학습시킬 때 사용한 이미지의 한 변의 길이. 정사각형이기 때문에 한 변의 길이만 받아도 된다. original_shape는 카메라에서 얻은 프레임의 너비, 높이가 담겨있다
    orig_h, orig_w = original_shape

    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))

    # Image height and width after padding is removed
    # 전처리 중에 padding이 있었기 때문에 padding 과정에서 외곽에 추가된 것들을 제외하는 연산
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x

    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes # 원본 이미지에 적합한 bbox들을 반환

# pytorchyolo/utils/utils.py의 non_max_suppression를 수정한 메소드
# YOLOv3에서 얻은 bbox는 [x_center, y_center, width, height, confidence, cls_preds]로 구성된다.
# 이를 사용하기 편하게 [x_center, y_center, width, height, confidence, 탐지한 객체의 index]로 바꿔주며 바꾸는 과정에서 NMS를 수행해 iou가 일정수치 이상인 bbox만 이용한다
def change_bbox_to_use(prediction, conf_thres=0.0): # prediction : [16, 10647, 8]
    # Settings
    max_nms = 2000  # maximum number of boxes into torchvision.ops.nms()

    for _, x in enumerate(prediction):  
        
        x = x[x[:, 4] > conf_thres]  # confidence
        
        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        i, j = (x[:, 5:] >= conf_thres).nonzero(as_tuple=False).T
        x = torch.cat( (box[i], x[i, j + 5, None], j[:, None].float()), 1)

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            # sort by confidence
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]
    return x
# ====================================================================
# pytorchyolo/utils/utils.py에 있던 메소드들 =============================
# ====================================================================

# =========================================================================
# pytorchyolo/utils/transforms.py에 있던 메소드들 =============================
# =========================================================================
# DEFAULT_TRANSFORMS를 사용하기 위해 여러 class들을 가져왔음

class ToTensor(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        img, boxes = data
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(img)

        bb_targets = torch.zeros((len(boxes), 6))
        bb_targets[:, 1:] = transforms.ToTensor()(boxes)

        return img, bb_targets

# 텐서로 변한 이미지 크기를 조정하는데 사용된다
class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        img, boxes = data
        img = F.interpolate(img.unsqueeze(0), size=self.size, mode="nearest").squeeze(0)
        return img, boxes

# AbsoluteLabels를 통해 실제 값으로 변한 [x_center, y_center, width, height]를 다시 0~1 사이의 값으로 만드는데 사용되는 class다 
class RelativeLabels(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        img, boxes = data
        h, w, _ = img.shape
        boxes[:, [1, 3]] /= w
        boxes[:, [2, 4]] /= h
        return img, boxes

# YOLOv3이 출력하는 bbox의 [x_center, y_center, width, height]는 모두 0~1사이의 값을 가지고 있다. 
# AbsoluteLabels는 bbox에 있는 [x_center, y_center, width, height]에 이미지 사이즈와 곱해 실제 값으로 만들어준다
class AbsoluteLabels(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        img, boxes = data
        h, w, _ = img.shape
        boxes[:, [1, 3]] *= w
        boxes[:, [2, 4]] *= h
        return img, boxes

# PadSquare의 부모 클래스
class ImgAug(object):
    def __init__(self, augmentations=[]):
        self.augmentations = augmentations

    def __call__(self, data):
        # Unpack data
        img, boxes = data

        # Convert xywh to xyxy
        boxes = np.array(boxes)
        boxes[:, 1:] = xywh2xyxy_np(boxes[:, 1:])

        # Convert bounding boxes to imgaug
        bounding_boxes = BoundingBoxesOnImage(
            [BoundingBox(*box[1:], label=box[0]) for box in boxes],
            shape=img.shape)

        # Apply augmentations
        img, bounding_boxes = self.augmentations(
            image=img,
            bounding_boxes=bounding_boxes)

        # Clip out of image boxes
        bounding_boxes = bounding_boxes.clip_out_of_image()

        # Convert bounding boxes back to numpy
        boxes = np.zeros((len(bounding_boxes), 5))
        for box_idx, box in enumerate(bounding_boxes):
            # Extract coordinates for unpadded + unscaled image
            x1 = box.x1
            y1 = box.y1
            x2 = box.x2
            y2 = box.y2

            # Returns (x, y, w, h)
            boxes[box_idx, 0] = box.label
            boxes[box_idx, 1] = ((x1 + x2) / 2)
            boxes[box_idx, 2] = ((y1 + y2) / 2)
            boxes[box_idx, 3] = (x2 - x1)
            boxes[box_idx, 4] = (y2 - y1)

        return img, boxes

# padding을 위한 class
class PadSquare(ImgAug):
    def __init__(self, ):
        self.augmentations = iaa.Sequential([
            iaa.PadToAspectRatio(
                1.0,
                position="center-center").to_deterministic()
        ])

# YOLOv3에 입력값으로 넣기 전에 수행하는 전처리들
DEFAULT_TRANSFORMS = transforms.Compose([
    AbsoluteLabels(),
    PadSquare(),
    RelativeLabels(),
    ToTensor(),
])
# =========================================================================
# pytorchyolo/utils/transforms.py에 있던 메소드들 =============================
# =========================================================================
