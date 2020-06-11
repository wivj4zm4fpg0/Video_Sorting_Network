import os
import random

import cv2
import numpy as np
import torch
from PIL import Image
from natsort import natsorted
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid

from video_train_loader import VideoTrainDataSet


def recursive_video_path_load(input_dir: str, depth: int = 2, data_list=None):
    if data_list is None:
        data_list = []
    for file_name in os.listdir(input_dir):
        file_name_path = os.path.join(input_dir, file_name)
        if os.path.isfile(file_name_path):
            continue
        if depth > 0:
            recursive_video_path_load(file_name_path, depth - 1, data_list)
        else:
            data_list.append((file_name_path, 0))  # 0はダミー
    return data_list


class VideoTimeComparisonDataSet(VideoTrainDataSet):  # video_train_loader.VideoTrainDataSetを継承

    def __init__(self, pre_processing: transforms.Compose = None, frame_num: int = 8, path_load: list = None,
                 random_crop_size: int = 224, interval_frame: int = 0):
        super().__init__(pre_processing, frame_num, path_load, random_crop_size)
        self.interval_len = interval_frame
        self.shuffle_list = list(range(frame_num))
        self.frame_num = frame_num
        self.frame_half = frame_num // 2

    # イテレートするときに実行されるメソッド．ここをオーバーライドする必要がある．
    def __getitem__(self, index: int) -> tuple:
        while True:
            frame_list = [os.path.join(self.data_list[index][0], frame) for frame in
                          natsorted(os.listdir(self.data_list[index][0]))]
            frame_list = [frame for frame in frame_list if '.jpg' in frame or '.png' in frame]
            video_len = len(frame_list)
            if self.frame_num < video_len:
                break
            else:
                index = random.randint(0, len(self.data_list))

        start_index = random.randint(0, video_len - self.frame_num)
        frame_indices = list(range(video_len))[start_index:start_index + self.frame_num:self.interval_len + 1]

        if random.randint(0, 1) == 0:
            label = torch.tensor(0)
            first_indices = frame_indices[0:self.frame_half]
            second_indices = frame_indices[self.frame_half:self.frame_num]
        else:
            label = torch.tensor(1)
            first_indices = frame_indices[self.frame_half:self.frame_num]
            second_indices = frame_indices[0:self.frame_half]

        # transformsの設定
        # self.pre_processing.transforms[0].set_degree()  # RandomRotationの回転角度を設定
        # RandomCropの設定を行う. 引数に画像サイズが必要なので最初のフレームを渡す
        self.pre_processing.transforms[0].set_param(Image.open(frame_list[0]))
        # RandomHorizontalFlipのソースコード参照．pの値を設定．0なら反転しない，1なら反転する
        self.pre_processing.transforms[1].p = random.randint(0, 1)

        pre_processing = lambda image_path: self.pre_processing(Image.open(image_path).convert('RGB'))

        first_video_tensor = torch.stack([pre_processing(frame_list[i]) for i in first_indices])
        second_video_tensor = torch.stack([pre_processing(frame_list[i]) for i in second_indices])

        return torch.cat([first_video_tensor, second_video_tensor]), label  # 入力画像とそのラベルをタプルとして返す

    def __len__(self) -> int:  # データセットの数を返すようにする
        return len(self.data_list)


if __name__ == '__main__':  # UCF101データセットの読み込みテストを行う

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=2, required=False)
    parser.add_argument('--depth', type=int, default=1, required=False)
    parser.add_argument('--frame_num', type=int, default=8, required=False)

    args = parser.parse_args()

    data_loader = DataLoader(
        VideoTimeComparisonDataSet(
            path_load=recursive_video_path_load(args.dataset_path, args.depth),
            interval_frame=0,
            random_crop_size=180
        ),
        batch_size=args.batch_size, shuffle=False
    )


    def image_show(img):  # 画像を表示
        np_img = np.transpose(make_grid(img).numpy(), (1, 2, 0))
        cv2.imshow('image', cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))
        cv2.moveWindow('image', 50, 200)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            exit(0)


    for data in data_loader:
        inputs, labels = data
        print(f'{labels = }')
        for images_per_batch in inputs:
            image_show(images_per_batch)
