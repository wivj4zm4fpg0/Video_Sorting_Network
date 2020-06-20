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


class VideoSiamese3DCNNTrainDataSet(VideoTrainDataSet):  # video_train_loader.VideoTrainDataSetを継承

    def __init__(self, pre_processing: transforms.Compose = None, ref_frame_num: int = 16, path_list: list = None,
                 random_crop_size: int = 224, search_frame_num: int = 64):
        super().__init__(pre_processing=pre_processing, path_list=path_list, random_crop_size=random_crop_size)
        assert search_frame_num % ref_frame_num == 0
        self.search_frame_num = search_frame_num
        self.ref_frame_num = ref_frame_num
        self.split_num = search_frame_num // ref_frame_num

    # イテレートするときに実行されるメソッド．ここをオーバーライドする必要がある．
    def __getitem__(self, index: int) -> tuple:
        while True:
            frame_list = [os.path.join(self.data_list[index][0], frame) for frame in
                          natsorted(os.listdir(self.data_list[index][0]))]
            frame_list = [frame for frame in frame_list if '.jpg' in frame or '.png' in frame]
            video_len = len(frame_list)
            if self.search_frame_num < video_len:
                break
            else:
                index = random.randint(0, self.__len__())

        search_start_index = random.randint(0, video_len - self.search_frame_num)
        search_frame_indices = list(range(video_len))[search_start_index:search_start_index + self.search_frame_num]
        label = random.randint(0, self.split_num - 1)
        ref_start_index = label * self.ref_frame_num

        # transformsの設定
        # self.pre_processing.transforms[0].set_degree()  # RandomRotationの回転角度を設定
        # RandomCropの設定を行う. 引数に画像サイズが必要なので最初のフレームを渡す
        self.pre_processing.transforms[0].set_param(Image.open(frame_list[0]))
        # RandomHorizontalFlipのソースコード参照．pの値を設定．0なら反転しない，1なら反転する
        # self.pre_processing.transforms[1].p = random.randint(0, 1)
        self.pre_processing.transforms[1].p = 0
        pre_processing = lambda image_path: self.pre_processing(Image.open(image_path).convert('RGB'))

        search_videos = [pre_processing(frame_list[frame_index]) for frame_index in search_frame_indices]
        ref_videos = search_videos[ref_start_index: ref_start_index + self.ref_frame_num]
        search_videos = torch.stack(search_videos)
        ref_videos = torch.stack(ref_videos)
        label = torch.tensor(label)

        return search_videos, ref_videos, label  # 入力画像とそのラベルをタプルとして返す

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


if __name__ == '__main__':  # UCF101データセットの読み込みテストを行う

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=2, required=False)
    parser.add_argument('--depth', type=int, default=1, required=False)
    parser.add_argument('--frame_num', type=int, default=4, required=False)

    args = parser.parse_args()

    data_loader = DataLoader(
        VideoSiamese3DCNNTrainDataSet(
            path_list=recursive_video_path_load(args.dataset_path, args.depth),
            random_crop_size=180,
        ),
        batch_size=args.batch_size, shuffle=False
    )


    def image_show(img):  # 画像を表示
        np_img = np.transpose(make_grid(img).numpy(), (1, 2, 0))
        np_img = cv2.resize(np_img, (700, 700))
        cv2.imshow('image', cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))
        cv2.moveWindow('image', 500, 30)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            exit(0)


    for inputs in data_loader:
        search_videos, ref_videos, labels = inputs
        print(f'{labels = }')
        for i in range(len(search_videos)):
            image_show(search_videos[i])
            image_show(ref_videos[i])
