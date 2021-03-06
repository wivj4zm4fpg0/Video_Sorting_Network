import itertools
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


class VideoSort3DCNNTrainDataSet(VideoTrainDataSet):  # video_train_loader.VideoTrainDataSetを継承

    def __init__(self, pre_processing: transforms.Compose = None, frame_num: int = 4, path_list: list = None,
                 random_crop_size: int = 224, cnn_frame_num: int = 16):
        super().__init__(pre_processing, frame_num, path_list, random_crop_size, frame_interval=0)
        assert cnn_frame_num % frame_num == 0
        self.cnn_frame_num = cnn_frame_num
        self.crop_frame_len = cnn_frame_num * frame_num
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
        while True:
            nat_list = natsorted(os.listdir(self.data_list[index][0]))
            frame_list = [os.path.join(self.data_list[index][0], frame) for frame in nat_list]
            frame_list = [frame for frame in frame_list if '.jpg' in frame or '.png' in frame]
            video_len = len(frame_list)
            if self.crop_frame_len < video_len:
                break
            else:
                index = random.randint(0, len(self.data_list))

        start_index = random.randint(0, video_len - self.crop_frame_len)
        frame_indices = list(range(video_len))[start_index:start_index + self.crop_frame_len:self.cnn_frame_num]
        frame_indices = [list(range(frame_index, frame_index + self.cnn_frame_num)) for frame_index in frame_indices]
        shuffle_frame_indices = list(range(self.frame_num))
        label = self.shuffle_list[random.randint(0, self.shuffle_len - 1)]
        for i, v in enumerate(label):
            shuffle_frame_indices[i] = (frame_indices[v])

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

        return output_videos, torch.tensor(label)  # 入力画像とそのラベルをタプルとして返す

    def __len__(self) -> int:  # データセットの数を返すようにする
        return len(self.data_list)


if __name__ == '__main__':  # UCF101データセットの読み込みテストを行う

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1, required=False)
    parser.add_argument('--depth', type=int, default=1, required=False)
    parser.add_argument('--frame_num', type=int, default=4, required=False)

    args = parser.parse_args()

    data_loader = DataLoader(
        VideoSort3DCNNTrainDataSet(
            path_list=recursive_video_path_load(args.dataset_path, args.depth),
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
