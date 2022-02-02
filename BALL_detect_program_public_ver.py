
from __future__ import division

# ====================================
# 전역변수 선언, 정의 =====================
# ====================================
# ball index
global RED_BALL_INDEX 
global WHITE_BALL_INDEX  
global YELLOW_BALL_INDEX 

RED_BALL_INDEX = 4
WHITE_BALL_INDEX = 5
YELLOW_BALL_INDEX= 6

# 공 색상
global white_color
global yellow_color
global red_color
white_color = (255, 255, 255)
yellow_color = (255,255,0)
red_color = (255, 0, 0)
# ====================================
# 전역변수 선언, 정의 =====================
# ====================================

import torch

import numpy as np
import cv2
from itertools import chain
import os

import torch.nn as nn
import torchvision
import torchvision.transforms
import torch.nn.functional
import time
import imgaug.augmenters
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

def filter_in_BiliardField(detect_ball_coordi_group) :

    # left-up 
    detect_ball_coordi_group = detect_ball_coordi_group[detect_ball_coordi_group[:,0] >= 10]
    detect_ball_coordi_group = detect_ball_coordi_group[detect_ball_coordi_group[:,1] >= 20]

    detect_ball_coordi_group = detect_ball_coordi_group[detect_ball_coordi_group[:,0] <= 1221]
    detect_ball_coordi_group = detect_ball_coordi_group[detect_ball_coordi_group[:,1] <= 635]


    return detect_ball_coordi_group

def detect_biliard_ball(model, image, device, window_name, BALLS, img_size = 416, nms_thres = 0.5) : # 입력받은 프레임을 가지고 공 탐지

    # 공들의 좌표를 출력하기 위해 사용하는 BALL 클래스의 객체들
    white_ball = BALLS[0]
    yellow_ball = BALLS[1]
    red_ball = BALLS[2]
    
    # 입력받은 프레임을 전처리
    input_img = torchvision.transforms.Compose([
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

    red_ball_coordi_group = filter_in_BiliardField(red_ball_coordi_group)
    white_ball_coordi_group = filter_in_BiliardField(white_ball_coordi_group)
    yellow_ball_coordi_group = filter_in_BiliardField(yellow_ball_coordi_group)

    # 따로 모은 것들에서 가장 confience가 높은 것을 하나씩 뽑음
    # 여기서 얻은 ball_coordi들을 YOLOv3가 탐지한 공들의 좌표로 사용한다
    if white_ball_coordi_group.size()[0] > 0 :
        white_ball_coordi = white_ball_coordi_group[white_ball_coordi_group[:,4].argmax(),:].numpy()
    else : 
        white_ball_coordi = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # 만약 공을 탐지하지 못했으면 모든 값을 0.0으로 설정

    if yellow_ball_coordi_group.size()[0] > 0 :
        yellow_ball_coordi = yellow_ball_coordi_group[yellow_ball_coordi_group[:,4].argmax(),:].numpy()
    else :
        yellow_ball_coordi = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    if red_ball_coordi_group.size()[0] > 0 :
        red_ball_coordi = red_ball_coordi_group[red_ball_coordi_group[:,4].argmax(),:].numpy()
    else :
        red_ball_coordi = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # ball_coordi들을 각 공의 좌표로 설정
    white_ball.set_coordi(white_ball_coordi)
    yellow_ball.set_coordi(yellow_ball_coordi)
    red_ball.set_coordi(red_ball_coordi)

    detections = [white_ball, yellow_ball, red_ball] # ball 객체들로 구성된 리스트 생성

    for detect_bbox in detections :
        # 공의 종류에 맞는 색을 가진 bbox를 출력
        if detect_bbox.color_idx == RED_BALL_INDEX : # 빨간공
            if detect_bbox.isValid() == True : # Ball이 가지고 있는 값들이 모두 0.0이 아니면 공의 좌표를 출력. 앞서 공을 탐지 못했을시 모든 값을 0.0으로 설정한 이유
                cv2.rectangle(image, (detect_bbox.min_x, detect_bbox.min_y), (detect_bbox.max_x, detect_bbox.max_y), red_color, 2)
        elif detect_bbox.color_idx == WHITE_BALL_INDEX : # 흰공
            if detect_bbox.isValid() == True :
                cv2.rectangle(image, (detect_bbox.min_x, detect_bbox.min_y), (detect_bbox.max_x, detect_bbox.max_y), white_color, 2)
        elif detect_bbox.color_idx == YELLOW_BALL_INDEX : # 노란공
            if detect_bbox.isValid() == True :
                cv2.rectangle(image, (detect_bbox.min_x, detect_bbox.min_y), (detect_bbox.max_x, detect_bbox.max_y), yellow_color, 2)
        
    cv2.imshow(window_name, image[:, :, [2, 1, 0]]) # RGB -> BGR변환. opencv는 BGR을 사용한다
    cv2.waitKey(1)

    return white_ball, yellow_ball, red_ball # 탐지한 공들의 좌표가 담긴 BALL 클래스의 객체들을 반환

# YOLOv3 구현코드의 pytorchyolo/utils/utils.py에 있는 메소드
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

# YOLOv3 구현코드의 pytorchyolo/utils/parse_config.py에 있는 메소드
def parse_model_config(path):
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
    module_defs = []
    for line in lines:
        if line.startswith('['):  # This marks the start of a new block
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs

def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)
    hyperparams.update({
        'batch': int(hyperparams['batch']),
        'subdivisions': int(hyperparams['subdivisions']),
        'width': int(hyperparams['width']),
        'height': int(hyperparams['height']),
        'channels': int(hyperparams['channels']),
        'optimizer': hyperparams.get('optimizer'),
        'momentum': float(hyperparams['momentum']),
        'decay': float(hyperparams['decay']),
        'learning_rate': float(hyperparams['learning_rate']),
        'burn_in': int(hyperparams['burn_in']),
        'max_batches': int(hyperparams['max_batches']),
        'policy': hyperparams['policy'],
        'lr_steps': list(zip(map(int,   hyperparams["steps"].split(",")),
                             map(float, hyperparams["scales"].split(","))))
    })
    assert hyperparams["height"] == hyperparams["width"], \
        "Height and width should be equal! Non square images are padded with zeros."
    output_filters = [hyperparams["channels"]]
    module_list = nn.ModuleList()
    for module_i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2
            modules.add_module(
                f"conv_{module_i}",
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias=not bn,
                ),
            )
            if bn:
                modules.add_module(f"batch_norm_{module_i}",
                                   nn.BatchNorm2d(filters, momentum=0.1, eps=1e-5))
            if module_def["activation"] == "leaky":
                modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))
            if module_def["activation"] == "mish":
                modules.add_module(f"mish_{module_i}", Mish())

        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                modules.add_module(f"_debug_padding_{module_i}", nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride,
                                   padding=int((kernel_size - 1) // 2))
            modules.add_module(f"maxpool_{module_i}", maxpool)

        elif module_def["type"] == "upsample":
            upsample = Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
            modules.add_module(f"upsample_{module_i}", upsample)

        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum([output_filters[1:][i] for i in layers]) // int(module_def.get("groups", 1))
            modules.add_module(f"route_{module_i}", nn.Sequential())

        elif module_def["type"] == "shortcut":
            filters = output_filters[1:][int(module_def["from"])]
            modules.add_module(f"shortcut_{module_i}", nn.Sequential())

        elif module_def["type"] == "yolo":
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # Extract anchors
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def["classes"])
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes)
            modules.add_module(f"yolo_{module_i}", yolo_layer)
        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list


class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x

class Mish(nn.Module):
    """ The MISH activation function (https://github.com/digantamisra98/Mish) """

    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))

class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes):
        super(YOLOLayer, self).__init__()
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.no = num_classes + 5  # number of outputs per anchor
        self.grid = torch.zeros(1)  # TODO

        anchors = torch.tensor(list(chain(*anchors))).float().view(-1, 2)
        self.register_buffer('anchors', anchors)
        self.register_buffer(
            'anchor_grid', anchors.clone().view(1, -1, 1, 1, 2))
        self.stride = None

    def forward(self, x, img_size):
        stride = img_size // x.size(2)
        self.stride = stride
        
#         print("before : " ,x.shape)
        
        bs, _, ny, nx = x.shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
        x = x.view(bs, self.num_anchors, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        
#         print("after : ", x.shape)

        if not self.training:  # inference
            if self.grid.shape[2:4] != x.shape[2:4]:
                self.grid = self._make_grid(nx, ny).to(x.device)

            x[..., 0:2] = (x[..., 0:2].sigmoid() + self.grid) * stride  # xy
            x[..., 2:4] = torch.exp(x[..., 2:4]) * self.anchor_grid # wh
            x[..., 4:] = x[..., 4:].sigmoid()
            x = x.view(bs, -1, self.no)

        return x

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, config_path):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(config_path)
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.yolo_layers = [layer[0]
                            for layer in self.module_list if isinstance(layer[0], YOLOLayer)]
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)

    def forward(self, x):
        img_size = x.size(2)
        layer_outputs, yolo_outputs = [], []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif module_def["type"] == "route":
                combined_outputs = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
                group_size = combined_outputs.shape[1] // int(module_def.get("groups", 1))
                group_id = int(module_def.get("group_id", 0))
                x = combined_outputs[:, group_size * group_id : group_size * (group_id + 1)] # Slice groupings used by yolo v4
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def["type"] == "yolo":
                x = module[0](x, img_size)
                yolo_outputs.append(x)
            layer_outputs.append(x)
            
#         print("self.training : ", self.training)
            
        return yolo_outputs if self.training else torch.cat(yolo_outputs, 1)

    def load_darknet_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        with open(weights_path, "rb") as f:
            # First five are header values
            header = np.fromfile(f, dtype=np.int32, count=5)
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Establish cutoff for loading backbone weights
        cutoff = None
        # If the weights file has a cutoff, we can find out about it by looking at the filename
        # examples: darknet53.conv.74 -> cutoff is 74
        filename = os.path.basename(weights_path)
        if ".conv." in filename:
            try:
                cutoff = int(filename.split(".")[-1])  # use last part of filename
            except ValueError:
                pass

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if i == cutoff:
                break
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(
                    weights[ptr: ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    def save_darknet_weights(self, path, cutoff=-1):
        """
            @:param path    - path of the new weights file
            @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        """
        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()


def load_model(model_path, weights_path=None):
    """Loads the yolo model from file.

    :param model_path: Path to model definition file (.cfg)
    :type model_path: str
    :param weights_path: Path to weights or checkpoint file (.weights or .pth)
    :type weights_path: str
    :return: Returns model
    :rtype: Darknet
    """
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")  # Select device for inference
    model = Darknet(model_path).to(device)

    model.apply(weights_init_normal)

    # If pretrained weights are specified, start from checkpoint or weight file
    if weights_path:
        if weights_path.endswith(".pth"):
            # Load checkpoint weights
            model.load_state_dict(torch.load(weights_path, map_location=device))
        else:
            # Load darknet weights
            model.load_darknet_weights(weights_path)
    return model


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
def change_bbox_to_use(prediction, iou_thres): # prediction : [16, 10647, 8]
    # Settings
    # (pixels) minimum and maximum box width and height
    max_wh = 4096
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    
    t = time.time()
    output = [torch.zeros((0, 6), device="cpu")] * prediction.shape[0] # [-1, 6]인 리스트. -

    for xi, x in enumerate(prediction):  
        
        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        i, j = (x[:, 5:] >= 0.1).nonzero(as_tuple=False).T
        x = torch.cat( (box[i], x[i, j + 5, None], j[:, None].float()), 1)

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            # sort by confidence
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]
            
        # Batched NMS
        c = x[:, 5:6] * max_wh  # classes
        # boxes (offset by class), scores
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS. iou가 iou_thres를 넘기는 bbox만 사용
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
            
        output[xi] = to_cpu(x[i])
    
    return output
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
        img = torchvision.transforms.ToTensor()(img)

        bb_targets = torch.zeros((len(boxes), 6))
        bb_targets[:, 1:] = torchvision.transforms.ToTensor()(boxes)

        return img, bb_targets

# 텐서로 변한 이미지 크기를 조정하는데 사용된다
class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        img, boxes = data
        img = torch.nn.functional.interpolate(img.unsqueeze(0), size=self.size, mode="nearest").squeeze(0)
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
        self.augmentations = imgaug.augmenters.Sequential([
            imgaug.augmenters.PadToAspectRatio(
                1.0,
                position="center-center").to_deterministic()
        ])

# YOLOv3에 입력값으로 넣기 전에 수행하는 전처리들
DEFAULT_TRANSFORMS = torchvision.transforms.Compose([
    AbsoluteLabels(),
    PadSquare(),
    RelativeLabels(),
    ToTensor(),
])
# =========================================================================
# pytorchyolo/utils/transforms.py에 있던 메소드들 =============================
# =========================================================================



# 공의 위치를 나타낼 때 사용하는 클래스
# 당구대에 공이 탐지되지 않을 경우 모든 값을 0.0으로 설정해 화면에 나타낼 수 없게 만들기 위해 제작
# YOLOv3에서 얻은 공의 bbox를 사용할 수 있게 처리 후 BALL에 저장한다
class BALL :
    def __init__(self) :
        # opencv에서 이미지에 도형 그릴 때 필요한 값들
        self.min_x = 0
        self.min_y = 0
        self.max_x = 0
        self.max_y = 0

        # 색깔의 index. YOLOv3의 출력값에 있다.
        self.color_idx = 0
        
    # bbox를 BALL에 저장하는 메소드
    def set_coordi(self, coordi) :
        coordi = coordi.astype(np.int32)

        self.min_x = coordi[0]
        self.min_y = coordi[1]
        self.max_x = coordi[2]
        self.max_y = coordi[3]
        self.color_idx = coordi[5]

    def isValid(self) : 
        return not (self.min_x == 0 & self.min_y == 0 & self.max_x == 0 & self.max_y == 0)

# YOLOv3을 불러오기 위한 메소드
def ready_for_detect() :
    # 모델의 구조가 담긴 cfg와 학습시킨 가중치들이 담긴 pth의 경로
    current_file_path = os.getcwd()
    model_path = current_file_path + '/yolov3/yolov3-for-Biliard-7Classes.cfg'
    pretrained_weights = current_file_path + '/yolov3/yolov3_weights_7Classes.pth'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 파이토치를 이용해 만든 객체, 변수들을 gpu에서 돌릴지 cpu에서 돌릴지 결정.
    model = load_model(model_path, pretrained_weights) # 학습시킨 가중치가 저장된 YOLOv3을 불러옴
    model.eval() # YOLOv3을 테스트 모드로 변경
    return model, device

def main() :
    model, device = ready_for_detect() # YOLOv3과 파이토치 기반 객체, 변수를 돌릴 장치(cpu or gpu)에 대한 정보를 휙득

    capture = cv2.VideoCapture(0) # 연결된 카메라 혹은 비디오에서 프레임을 얻기 위해 필요한 객체. 0을 매개변수로 전달하면 연결된 카메라에서 프레임을 얻어온다

    # 캡처하는 프레임의 해상도 조정
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # 공의 위치정보를 담을 BALL 클래스의 객체들을 정의
    white_ball = BALL()
    yellow_ball = BALL()
    red_ball = BALL()

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
        white_ball, yellow_ball, red_ball = detect_biliard_ball(model, frame.copy(), device, 'Detect_ball', [white_ball, yellow_ball, red_ball]) # 공 탐지 후 화면에 보여줌
        
    capture.release() # 캡처 중지
    cv2.destroyAllWindows() # 화면 생성 중단

if __name__ == "__main__":
    main()
