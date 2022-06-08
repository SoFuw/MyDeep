from ssl import Purpose
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import os
import pickle
import torch.nn.functional as F
import json


class KeyPointDataset(Dataset):

    def __init__(self, input_sequence: int, purpose: str) -> None:
        super().__init__()
        self.input_sequence = input_sequence
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

                keydata = data["pred_keypoints"].view(-1, 51)
                keydata = torch.squeeze(keydata).squeeze()
                class_data = csv_data.values[index-1][1]
                buf_data[index-1] = [keydata, class_data]

            for i in range(len(buf_data)):
                if(buf_data[i] == None):
                    continue
                filtered_data.append(buf_data[i])

        for target_class in [1, 2, 3, 4, 5, 6]:
            datadict = {}
            datadict['class'] = target_class
            datadict['data'] = []
            self.data.append(datadict)

        for data in filtered_data:
            #print(data[1]-1)
            self.data[data[1]-1]['data'].append(data[0])

        length = 0
        for mini_data in self.data:
            left = length
            length += len(mini_data['data'])-(self.input_sequence-1)
            right = length-1
            self.data_range.append((left, right))

        self.len = length

    def __len__(self) -> int:
        return self.len

    def __getclass__(self, idx):
        for target_class, (left, right) in enumerate(self.data_range):
            if (idx >= left and idx <= right):
                return idx-left, target_class

    def __getitem__(self, idx):
        i, target_class = self.__getclass__(idx)
        key_data = self.data[target_class]['data'][i:i+self.input_sequence]
        key_data = torch.stack(key_data, dim=0)
        target_class = F.one_hot(torch.tensor(target_class), num_classes=6)
        target_class = target_class.type(torch.float32)
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
