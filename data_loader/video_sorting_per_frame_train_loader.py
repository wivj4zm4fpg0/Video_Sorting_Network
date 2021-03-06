"""
途中だけ並び替える
1, 2, 3, 4, [5, 6, 7, 8], 9, 10, 11, 12
1, 2, 3, 4, [7, 6, 8, 5], 9, 10, 11, 12
"""
import os
import random

import cv2
import numpy as np
import torch
from PIL import Image
from natsort import natsorted
from torch.utils.data import DataLoader
from torchvision import transforms
import copy
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


class VideoSortingPerFrameTrainDataSet(VideoTrainDataSet):  # video_train_loader.VideoTrainDataSetを継承

    def __init__(self, pre_processing: transforms.Compose = None, frame_num=12, path_list: list = None,
                 random_crop_size=224, interval_frame=4, sorting_start_index=4, sorting_end_index=8):
        super().__init__(pre_processing, frame_num, path_list, random_crop_size)
        self.crop_video_len = (frame_num - 1) * interval_frame + frame_num
        self.interval_len = interval_frame
        self.shuffle_list = list(range(sorting_start_index, sorting_end_index))
        self.sorting_start_index = sorting_start_index

    # イテレートするときに実行されるメソッド．ここをオーバーライドする必要がある．
    def __getitem__(self, index: int) -> tuple:
        frame_list = \
            [os.path.join(self.data_list[index][0], frame) for frame in natsorted(os.listdir(self.data_list[index][0]))]
        frame_list = [frame for frame in frame_list if '.jpg' in frame or '.png' in frame]
        video_len = len(frame_list)
        assert self.crop_video_len < video_len
        # {frame_index + 0, frame_index + 1, ..., frame_index + self.frame_num - 1}番号のフレームを取得するのに使う
        start_index = random.randint(0, video_len - self.crop_video_len)
        frame_indices = list(range(video_len))[start_index:start_index + self.crop_video_len:self.interval_len + 1]
        frame_indices = [[frame_indices[i], i] for i in range(self.frame_num)]
        shuffle_indices = copy.copy(frame_indices)
        random.shuffle(self.shuffle_list)
        for i, e in enumerate(self.shuffle_list):
            shuffle_indices[i + self.sorting_start_index] = frame_indices[e]
        shuffle_indices = torch.tensor(shuffle_indices)


        # transformsの設定
        # self.pre_processing.transforms[0].set_degree()  # RandomRotationの回転角度を設定
        # RandomCropの設定を行う. 引数に画像サイズが必要なので最初のフレームを渡す
        self.pre_processing.transforms[0].set_param(Image.open(frame_list[0]))
        # RandomHorizontalFlipのソースコード参照．pの値を設定．0なら反転しない，1なら反転する
        self.pre_processing.transforms[1].p = random.randint(0, 1)

        pre_processing = lambda image_path: self.pre_processing(Image.open(image_path).convert('RGB'))
        # リスト内包表記で検索

        # video_tensor = [pre_processing(frame_list[i]) for i in frame_indices]
        video_tensor = [pre_processing(frame_list[int(i)]) for i in shuffle_indices[:, 0]]
        video_tensor = torch.stack(video_tensor)  # 3次元Tensorを含んだList -> 4次元Tensorに変換

        return video_tensor, shuffle_indices[:, 1]  # 入力画像とそのラベルをタプルとして返す

    def __len__(self) -> int:  # データセットの数を返すようにする
        return len(self.data_list)

    def update_shuffle_list(self):
        self.shuffle_list = random.sample(self.shuffle_list, self.frame_num)

    def frames_shuffle(self, input_tensor: torch.Tensor) -> torch.Tensor:
        shuffle_input = torch.zeros(input_tensor.size())
        for i, frames in enumerate(input_tensor):
            for j, shuffle_value in enumerate(self.shuffle_list):
                shuffle_input[i][j] = input_tensor[i][shuffle_value]
        return shuffle_input

    def shuffle(self, inputs: torch.Tensor, labels: torch.Tensor) -> tuple:
        shuffle_list = list(range(self.frame_num))
        random.shuffle(shuffle_list)
        new_labels = []
        new_inputs = []
        for i in range(len(inputs)):
            new_seq = []
            new_label = []
            for shuffle_value in shuffle_list:
                new_label.append(labels[i][shuffle_value])
                new_seq.append(inputs[i][shuffle_value])
            new_labels.append(torch.stack(new_label))
            new_inputs.append(torch.stack(new_seq))
        new_labels = torch.stack(new_labels)
        new_inputs = torch.stack(new_inputs)
        return new_inputs, new_labels


if __name__ == '__main__':  # UCF101データセットの読み込みテストを行う

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=3, required=False)
    parser.add_argument('--depth', type=int, default=1, required=False)
    parser.add_argument('--frame_num', type=int, default=4, required=False)
    parser.add_argument('--interval_frame', type=int, default=1, required=False)

    args = parser.parse_args()

    data_loader = DataLoader(
        VideoSortingPerFrameTrainDataSet(
            path_list=recursive_video_path_load(args.dataset_path, args.depth),
            interval_frame=args.interval_frame,
            random_crop_size=180
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
        inputs, labels = data
        print(f'{labels = }')
        for images_per_batch in inputs:
            image_show(images_per_batch)
