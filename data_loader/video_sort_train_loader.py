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


class VideoSortTrainDataSet(VideoTrainDataSet):  # video_train_loader.VideoTrainDataSetを継承

    def __init__(self, pre_processing: transforms.Compose = None, frame_num = 4, path_list: list = None,
                 random_crop_size = 224, frame_interval = 1):
        super().__init__(pre_processing, frame_num, path_list, random_crop_size, frame_interval=frame_interval)
        # a,b,c,dとd,c,b,aを同一とみなすことにしてn!/2の順列を作成
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

        nat_list = natsorted(os.listdir(self.data_list[index][0]))
        frame_list = [os.path.join(self.data_list[index][0], frame) for frame in nat_list]
        frame_list = [frame for frame in frame_list if '.jpg' in frame or '.png' in frame]
        video_len = len(frame_list)
        assert self.crop_video_len < video_len

        start_index = random.randint(0, video_len - self.crop_video_len)
        frame_indices = list(range(video_len))[start_index:start_index + self.crop_video_len:self.frame_interval + 1]
        shuffle_frame_indices = list(range(self.frame_num))
        label = self.shuffle_list[random.randint(0, self.shuffle_len - 1)]
        for i, v in enumerate(label):
            shuffle_frame_indices[i] = frame_indices[v]

        # transformsの設定
        # self.pre_processing.transforms[0].set_degree()  # RandomRotationの回転角度を設定
        # RandomCropの設定を行う. 引数に画像サイズが必要なので最初のフレームを渡す
        self.pre_processing.transforms[0].set_param(Image.open(frame_list[0]))
        # RandomHorizontalFlipのソースコード参照．pの値を設定．0なら反転しない，1なら反転する
        self.pre_processing.transforms[1].p = random.randint(0, 1)

        pre_processing = lambda image_path: self.pre_processing(Image.open(image_path).convert('RGB'))
        video_tensor = [pre_processing(frame_list[int(i)]) for i in shuffle_frame_indices]
        video_tensor = torch.stack(video_tensor)  # 3次元Tensorを含んだList -> 4次元Tensorに変換

        return video_tensor, torch.tensor(label)  # 入力画像とそのラベルをタプルとして返す

    def __len__(self) -> int:  # データセットの数を返すようにする
        return len(self.data_list)


if __name__ == '__main__':  # UCF101データセットの読み込みテストを行う

    import argparse
    from data_loader.path_list_loader import recursive_video_path_load

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=3, required=False)
    parser.add_argument('--depth', type=int, default=1, required=False)
    parser.add_argument('--frame_num', type=int, default=4, required=False)
    parser.add_argument('--interval_frame', type=int, default=1, required=False)

    args = parser.parse_args()

    data_loader = DataLoader(
        VideoSortTrainDataSet(
            path_list=recursive_video_path_load(args.dataset_path, args.depth),
            frame_interval=args.interval_frame,
            random_crop_size=180
        ),
        batch_size=args.batch_size, shuffle=False
    )


    def image_show(img):  # 画像を表示
        np_img = np.transpose(make_grid(img).numpy(), (1, 2, 0))
        cv2.imshow('image', cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))
        cv2.moveWindow('image', 100, 200)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            exit(0)


    for data in data_loader:
        inputs, labels = data
        print(f'{labels = }')
        for images_per_batch in inputs:
            image_show(images_per_batch)
