from ssl import Purpose
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import os
import pickle
import torch.nn.functional as F
import json


class KeyPointDataset(Dataset):

    def __init__(self, purpose: str) -> None:
        super().__init__()
        self.data = []
        self.data_range = []

        with open("/home/fuw34/anaconda3/envs/myproject/myproject/data/dataInfo.json", "r") as f:
            data_info = json.load(f)

        filtered_data = []

        for target_name in data_info[purpose]:
            buf_data = [None]*len(data_info[purpose])
            target_dir = f'/home/fuw34/anaconda3/envs/myproject/myproject/data/{purpose}/{target_name}'
            keypoint_path = self.__get_all_file_path([target_dir], ["pkl"])
            class_path = self.__get_all_file_path([target_dir], ["csv"])
            csv_path = next(iter(class_path))
            pkl_path = next(iter(keypoint_path))
            csv_data = pd.read_csv(class_path[csv_path][0])

            with open(keypoint_path[pkl_path][0], "rb") as f:
                pkl_data = pickle.load(f)

            #좌우 반전 데이터 판독
            index = len(csv_data)
            if(index != len(pkl_data)):
                for iterdata in csv_data.values:
                    #print(iterdata)
                    csv_data.loc[index] = [index+1, iterdata[1]]
                    index = index+1

            buf_data = [None]*len(pkl_data)

            for data in pkl_data:
                index = data["name"]
                if("rgbref" in index):
                    continue
                index = index[4:-4]

                index = int(index)
                if(data["score"] == 0):
                    buf_data[index-1] = None
                    continue
                keydata = data["pred_keypoints"]

                class_data = csv_data.values[index-1][1]
                buf_data[index-1] = [keydata, class_data]

            for i in range(len(buf_data)):
                if(buf_data[i] == None):
                    continue
                elif(buf_data[i][1] == 6):
                    continue
                filtered_data.append(buf_data[i])

        self.data = filtered_data
        self.len = len(filtered_data)

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, idx: int):
        target_class = self.data[idx][1]-1
        target_class = F.one_hot(torch.tensor(target_class), num_classes=5)
        target_class = target_class.type(torch.float32)
        key_data = self.data[idx][0]

        return key_data, target_class

    def __get_last_dir(self, absolute_dir_path: str, fileExtensions: list) -> list:
        absolute_path = absolute_dir_path
        dirs = os.listdir(absolute_path)
        saved_dirs = []

        for target in dirs:
            target_path = absolute_path+"/"+target
            if os.path.isdir(target_path):
                saved_dirs.extend(self.__get_last_dir(target_path))

        for target in dirs:
            for_break = False
            for image_format in fileExtensions:
                if(image_format in target):
                    saved_dirs.append(absolute_dir_path)
                    for_break = True
                    break
            if(for_break):
                break
        return saved_dirs

    def __get_all_file_path(self, dir_list: list, fileExtensions: list) -> dict:
        images = {}
        for target_path in dir_list:
            images[target_path] = []
            target_dir = os.listdir(target_path)

            for target_image in target_dir:

                for images_file in fileExtensions:
                    if(images_file in target_image):
                        images[target_path].append(
                            target_path+"/"+target_image)
                        break

        return images
