import os
import random

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from natsort import natsorted
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid

from data_loader.video_train_loader import VideoTrainDataSet


def ucf101_sort_train_path_load(input_dir: str, input_label: str, train_label) -> list:
    train_list = []
    with open(train_label) as f:
        label_path_list = [s.strip() for s in f.readlines()]
        for label in label_path_list:
            label = os.path.basename(label)
            split_label = label.split(' ')
            train_list.append(split_label[0][:-4])

    path_list = []
    input_csv = pd.read_csv(input_label, engine='python')
    name_list = list(input_csv['name'])
    print(f'{len(train_list) = }')
    print(f'{len(name_list) = }')
    count = 0
    for class_ in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_)
        for video in os.listdir(class_path):
            if video not in name_list or video not in train_list:
                count += 1
                continue
            video_path = os.path.join(class_path, video)
            label = input_csv.loc[input_csv.name == video, 'label1':'order'].values[0]
            path_list.append((video_path, label[:-1], label[-1]))
    print(f'{count = }')
    return path_list


def load_sort_label(input_dir: str, input_label: str):
    path_list = []
    input_csv = pd.read_csv(input_label, engine='python')
    for class_ in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_)
        for video in os.listdir(class_path):
            video_path = os.path.join(class_path, video)
            label = input_csv.loc[input_csv.name == video, 'label1':'order'].values[0]
            path_list.append((video_path, label[:-1], label[-1]))
    return path_list


class VideoSortLabel3DCNNTrainDataSet(VideoTrainDataSet):  # video_train_loader.VideoTrainDataSetを継承

    def __init__(self, pre_processing: transforms.Compose = None, frame_num: int = 4, path_list: list = None,
                 random_crop_size: int = 224, cnn_frame_num: int = 8):
        super().__init__(pre_processing, frame_num, path_list, random_crop_size, frame_interval=0)
        self.cnn_frame_num = cnn_frame_num
        self.crop_frame_len = cnn_frame_num * frame_num

    # イテレートするときに実行されるメソッド．ここをオーバーライドする必要がある．
    def __getitem__(self, index: int) -> tuple:
        while True:
            nat_list = natsorted(os.listdir(self.data_list[index][0]))
            frame_list = [os.path.join(self.data_list[index][0], frame) for frame in nat_list]
            frame_list = [frame for frame in frame_list if '.jpg' in frame or '.png' in frame]
            video_len = len(frame_list)
            if self.crop_frame_len < video_len:
                break
            else:
                index = random.randint(0, len(self.data_list))

        frame_indices = list(range(video_len))
        label = self.data_list[index][1]
        shuffle_frame_indices = []
        for v in label:
            label_indices = v.split('_')
            start_index = int(label_indices[0])
            end_index = int(label_indices[1])
            shuffle_frame_indices.append(frame_indices[start_index:end_index])

        # transformsの設定
        # self.pre_processing.transforms[0].set_degree()  # RandomRotationの回転角度を設定
        # RandomCropの設定を行う. 引数に画像サイズが必要なので最初のフレームを渡す
        self.pre_processing.transforms[0].set_param(Image.open(frame_list[0]))
        # RandomHorizontalFlipのソースコード参照．pの値を設定．0なら反転しない，1なら反転する
        # self.pre_processing.transforms[1].p = random.randint(0, 1)
        self.pre_processing.transforms[1].p = 0
        pre_processing = lambda image_path: self.pre_processing(Image.open(image_path).convert('RGB'))

        output_videos = []
        for frames in shuffle_frame_indices:
            video_tensor = [pre_processing(frame_list[frame_index]) for frame_index in frames]
            video_tensor = torch.stack(video_tensor)
            output_videos.append(video_tensor)
        output_videos = torch.stack(output_videos)

        # 入力画像とそのラベルをタプルとして返す
        return output_videos, torch.tensor([int(i) for i in self.data_list[index][2].split('_')])

    def __len__(self) -> int:  # データセットの数を返すようにする
        return len(self.data_list)


if __name__ == '__main__':  # UCF101データセットの読み込みテストを行う

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1, required=False)
    parser.add_argument('--frame_num', type=int, default=4, required=False)
    parser.add_argument('--input_csv', type=str, required=True)

    args = parser.parse_args()

    data_loader = DataLoader(
        VideoSortLabel3DCNNTrainDataSet(
            path_list=load_sort_label(args.dataset_path, args.input_csv),
            random_crop_size=180,
        ),
        batch_size=args.batch_size, shuffle=False
    )


    def image_show(img):  # 画像を表示
        np_img = np.transpose(make_grid(img).numpy(), (1, 2, 0))
        cv2.imshow('image', cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))
        cv2.moveWindow('image', 30, 100)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            exit(0)


    for data in data_loader:
        videos, labels = data
        print(f'{labels = }')
        for videos_per_batch in videos:
            for video_per_seq in videos_per_batch:
                image_show(video_per_seq)
