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

from data_loader.video_train_loader import VideoTrainDataSet


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


class VideoSortTestDataSet(VideoTrainDataSet):  # video_train_loader.VideoTrainDataSetを継承

    def __init__(self, pre_processing: transforms.Compose = None, frame_num: int = 4, path_list: list = None,
                 random_crop_size: int = 224, interval_frame: int = 4):
        super().__init__(pre_processing, frame_num, path_list, random_crop_size)
        self.crop_video_len = (frame_num - 1) * interval_frame + frame_num
        self.interval_len = interval_frame

    # イテレートするときに実行されるメソッド．ここをオーバーライドする必要がある．
    def __getitem__(self, index: int) -> tuple:
        # print(f'{self.data_list = }')
        # print(f'{self.data_list[index] = }')
        frame_list = \
            [os.path.join(self.data_list[index][0], frame) for frame in natsorted(os.listdir(self.data_list[index][0]))]
        frame_list = [frame for frame in frame_list if '.jpg' in frame or '.png' in frame]
        video_len = len(frame_list)
        assert self.crop_video_len < video_len
        # {frame_index + 0, frame_index + 1, ..., frame_index + self.frame_num - 1}番号のフレームを取得するのに使う
        start_index = random.randint(0, video_len - self.crop_video_len)
        frame_indices = list(range(video_len))[start_index:start_index + self.crop_video_len:self.interval_len + 1]
        frame_indices = [[frame_indices[i], i] for i in range(self.frame_num)]
        shuffle_list = list(range(self.frame_num))
        shuffle_list = random.sample(shuffle_list, self.frame_num)
        shuffle_frame_indices = list(range(self.frame_num))
        for i, shuffle_value in enumerate(shuffle_list):
            shuffle_frame_indices[i] = frame_indices[shuffle_value]
        shuffle_frame_indices = torch.tensor(shuffle_frame_indices)

        # transformsの設定
        # self.pre_processing.transforms[0].set_degree()  # RandomRotationの回転角度を設定
        # RandomCropの設定を行う. 引数に画像サイズが必要なので最初のフレームを渡す
        self.pre_processing.transforms[0].set_param(Image.open(frame_list[0]))
        # RandomHorizontalFlipのソースコード参照．pの値を設定．0なら反転しない，1なら反転する
        self.pre_processing.transforms[1].p = random.randint(0, 1)

        pre_processing = lambda image_path: self.pre_processing(Image.open(image_path).convert('RGB'))
        # リスト内包表記で検索
        video_tensor = [pre_processing(frame_list[int(i)]) for i in shuffle_frame_indices[:, 0]]
        video_tensor = torch.stack(video_tensor)  # 3次元Tensorを含んだList -> 4次元Tensorに変換
        return video_tensor, shuffle_frame_indices[:, 1]  # 入力画像とそのラベルをタプルとして返す

    def __len__(self) -> int:  # データセットの数を返すようにする
        return len(self.data_list)


if __name__ == '__main__':  # UCF101データセットの読み込みテストを行う

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=3, required=False)
    parser.add_argument('--depth', type=int, default=1, required=False)
    parser.add_argument('--frame_num', type=int, default=4, required=False)
    parser.add_argument('--interval_frames', type=int, default=4, required=False)

    args = parser.parse_args()

    data_loader = DataLoader(
        VideoSortTestDataSet(
            path_list=recursive_video_path_load(args.dataset_path, args.depth),
            interval_frame=args.interval_frames,
            frame_num=args.frame_num
        ),
        batch_size=args.batch_size, shuffle=False
    )


    def image_show(img):  # 画像を表示
        np_img = np.transpose(make_grid(img).numpy(), (1, 2, 0))
        cv2.imshow('image', cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))
        cv2.moveWindow('image', 100, 200)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            exit(0)


    for input_images, labels in data_loader:
        print(f'{labels = }')
        for images_per_batch in input_images:
            image_show(images_per_batch)
