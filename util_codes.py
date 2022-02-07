
# 프로그램 작성 중에 유용했던 메소드들을 정리한 파이썬 파일입니다.
# 제작자 : 김민규(minkyu4506@gmail.com)
# 제작일 : 2021.2.3

# labelme로 만든 라벨 데이터를 프로그램에 사용한 YOLOv3의 라벨 데이터 양식으로 변환시키는 메소드
# 매개변수 : dataset_root_folder_path(데이터셋의 경로), ori_img_size(원래 이미지 크기를 저장한 리스트. 만약 원래 이미지가 1280 x 720이면 [1280, 720]이 ori_img_size가 됩니다.)
# ==============================데이터셋 구조============================== 
# dataset_root_folder
# ┗images(YOLOv3가 사용할 이미지들이 들어있는 폴더. 빈폴더여야 합니다.)
# ┗labels(YOLOv3가 사용할 라벨 데이터들이 들어있는 폴더. 빈폴더여야 합니다.)
# ┗ori_images(갖고 계시는 데이터셋의 이미지들이 들어있는 폴더.)
# ┗ori_labels(갖고 계시는 데이터셋의 라벨 데이터들이 들어있는 폴더.)
# Note : 이미지 파일의 이름과 라벨 데이터의 이름은 같아야 합니다.(예 : 000001.jpg의 라벨 데이터는 000001.json)
# ==============================데이터셋 구조==============================
import os 
import json
from PIL import Image
from tqdm import tqdm
import numpy as np

import torch
import torchvision.transforms as transforms

def convert_labelmeAugFile_to_YOLOv3_AugFile(dataset_root_folder_path) : 
    
    dataset_root_folder_path +='/' # 폴더명 -> 폴더명/ 으로 만들어줍니다. 예 : test/test_folder -> test/test_folder/

    # 폴더 경로 지정
    ori_image_folder = dataset_root_folder_path + 'ori_images'
    new_image_folder = dataset_root_folder_path + 'images'
    
    ori_annotation_folder = dataset_root_folder_path + 'ori_labels'
    new_annotation_folder = dataset_root_folder_path + 'labels'

    name_classes = [] # 갖고 계시는 데이터셋에 존재하는 클래스들의 이름을 저장하는 리스트

    ori_annotation_list = sorted(os.listdir(ori_annotation_folder)) # ori_labels에 있는 파일들의 이름이 저장된 리스트를 휙득

    # '.DS_Store'가 있는 경우가 있습니다. .DS_Store는 필요없는 파일이기 때문에 리스트에서 제거해줍니다.
    if '.DS_Store' in ori_annotation_list : 
        ori_annotation_list.remove('.DS_Store')
    
    for i in tqdm(range(0, len(ori_annotation_list)), desc = "make dataset for YOLOv3") :
        
        # ====================================이미지 사이즈를 416 x 416으로 변형 후 저장====================================
        img = Image.open(ori_image_folder + '/' + ori_annotation_list[i][:-5] + ".jpg").convert('RGB')
        img_size = img.size # 이미지의 원래 크기. bbox 변형에 사용합니다.

        img_resize = img.resize((416, 416))
        img_resize.save(new_image_folder + '/' + ori_annotation_list[i][:-5] + ".jpg")
        # ====================================이미지 사이즈를 416 x 416으로 변형 후 저장====================================

        # ============================================새로운 라벨 데이터 생성============================================
        # 프로그램에 사용한 YOLOv3는 라벨 데이터로 txt파일을 받기 때문에 txt파일을 생성합니다.
        new_txt_file = ori_annotation_list[i][:-5] + ".txt" 
        new_label_file = open(new_annotation_folder + "/" + new_txt_file, 'w')

        with open(ori_annotation_folder + '/' + ori_annotation_list[i], 'r') as f:
            json_data = json.load(f) # json file을 불러옵니다

        # bbox를 416 x 416 이미지 크기에 맞게 변형한 뒤 0~1 사이의 소수로 만들어줍니다.
        for annotation in json_data['shapes'] :
            label_idx = name_classes.index(annotation['label'])
                    
            x1 = annotation['points'][0][0] * (416.0/float(img_size[0]))
            y1 = annotation['points'][0][1] * (416.0/float(img_size[1]))

            x2 = annotation['points'][1][0] * (416.0/float(img_size[0]))
            y2 = annotation['points'][1][1] * (416.0/float(img_size[1]))

            x_center = ((x1 + x2) / 2) / 416.0
            y_center = ((y1 + y2) / 2) / 416.0
            width = (x2 - x1) / 416.0
            height = (y2 - y1) / 416.0

            new_annotation = str(label_idx) + " " + str(x_center) + " " + str(y_center) + " " + str(width) + " " + str(height) + "\n"
               
            new_label_file.write(new_annotation)

        new_label_file.close()
        # ============================================새로운 라벨 데이터 생성============================================


        # 하나의 데이터셋을 학습용, 검증용 데이터셋으로 나눕니다. train.txt에는 학습에 사용할 이미지의 경로를, valid.txt에는 검증에 사용할 이미지의 경로를 기록합니다. 
        # 이미지 파일의 이름과 라벨 데이터의 이름은 같기 때문에 이미지 파일의 경로만 저장해도 라벨 데이터를 불러올 수 있습니다.
        img_path_list = sorted(os.listdir(new_image_folder))

        if '.DS_Store' in img_path_list : 
            img_path_list.remove('.DS_Store')
        
        count = 1
        valid_dataset_start_index = int(len(img_path_list) * 0.75) # 몇번째 이미지부터 validation dataset으로 사용할 것인가(801이면 800번째 사진까지 학습용 데이터셋으로 사용)
        
        train_img_path_list_file = open(dataset_root_folder_path + 'train.txt', 'w')
        valid_img_path_list_file = open(dataset_root_folder_path + 'valid.txt', 'w')
        
        for img_name in img_path_list :
            img_path = dataset_root_folder_path + 'images' + '/' + img_name
            
            if count < valid_dataset_start_index :
                train_img_path_list_file.write(img_path + "\n")
            else : 
                valid_img_path_list_file.write(img_path + "\n")
            
            count+=1
        
        train_img_path_list_file.close()
        valid_img_path_list_file.close()


# YOLOv3은 (1,3,416,416) 크기의 tensor를 입력받습니다.
# 그래서 이미지 파일을 tensor로 만들어줘야 하며 make_img_to_input()이 그 역할을 수행합니다.
# 매개변수 : img_path(이미지 경로), device(tensor를 저장할 위치)
def make_img_to_input(img_path, device) :
    to_tensor = transforms.ToTensor()
    img = np.array(Image.open(img_path).convert('RGB').resize((416, 416)), dtype=np.uint8)
    img_tensor = to_tensor(img)
    img_tensor = torch.unsqueeze(img_tensor, 0).to(device)

    return img_tensor

