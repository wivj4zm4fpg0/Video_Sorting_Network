import itertools
import os
import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid

from data_loader.video_train_loader import VideoTrainDataSet

import scipy.io


def mat_loader(input_dir, input_mat):
    path_list = []
    mat = scipy.io.loadmat(input_mat, squeeze_me=True)
    for i in range(len(mat['filename'])):
        path_list.append((os.path.join(input_dir, mat['filename'][i]), mat['frame'][i]))
    return path_list


class VideoSortingClassificationMatTrainDataSet(VideoTrainDataSet):  # video_train_loader.VideoTrainDataSetを継承

    def __init__(self, pre_processing: transforms.Compose = None, frame_num: int = 4, path_list: list = None,
                 random_crop_size: int = 224, frame_interval: int = 4):
        super().__init__(pre_processing, frame_num, path_list, random_crop_size, frame_interval=frame_interval)
        sort_seq = list(itertools.permutations(list(range(frame_num)), frame_num))
        self.shuffle_list = []
        for v in sort_seq:
            v = list(v)
            if v[::-1] in self.shuffle_list:
                continue
            self.shuffle_list.append(v)
        self.shuffle_len = len(self.shuffle_list)

    # イテレートするときに実行されるメソッド．ここをオーバーライドする必要がある．
    def __getitem__(self, index: int) -> tuple:
        frame_list, video_len = self.getitem_init(index)
        assert self.crop_video_len < video_len

        frame_indices = self.data_list[index][1]

        shuffle_frame_indices = list(range(self.frame_num))
        label = random.randint(0, self.shuffle_len - 1)
        shuffle_list = self.shuffle_list[label]
        for i, v in enumerate(shuffle_list):
            shuffle_frame_indices[i] = frame_indices[v] // 2

        pre_processing = lambda image_path: self.pre_processing(Image.open(image_path).convert('RGB'))
        video_tensor = [pre_processing(frame_list[i]) for i in shuffle_frame_indices]
        video_tensor = torch.stack(video_tensor)  # 3次元Tensorを含んだList -> 4次元Tensorに変換

        return video_tensor, torch.tensor(label)  # 入力画像とそのラベルをタプルとして返す

    def __len__(self) -> int:  # データセットの数を返すようにする
        return len(self.data_list)


if __name__ == '__main__':  # UCF101データセットの読み込みテストを行う

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=4, required=False)
    parser.add_argument('--input_mat', type=str, required=False)

    args = parser.parse_args()

    data_loader = DataLoader(
        VideoSortingClassificationMatTrainDataSet(
            path_list=mat_loader(args.input_dir, args.input_mat),
        ),
        batch_size=args.batch_size, shuffle=False
    )


    def image_show(img):  # 画像を表示
        np_img = np.transpose(make_grid(img).numpy(), (1, 2, 0))
        cv2.imshow('image', cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))
        cv2.moveWindow('image', 30, 50)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            exit(0)

    print(f'{data_loader.dataset.shuffle_list = }')

    for data in data_loader:
        inputs, labels = data
        print(f'{labels = }')
        for images_per_batch in inputs:
            image_show(images_per_batch)
