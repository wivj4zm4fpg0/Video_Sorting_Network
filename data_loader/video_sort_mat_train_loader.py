import itertools
import os
import random

import cv2
import numpy as np
import scipy.io
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid

from data_loader.video_train_loader import VideoTrainDataSet


def mat_loader(input_dir, input_mat):
    path_list = []
    mat = scipy.io.loadmat(input_mat, squeeze_me=True)
    for i in range(len(mat['filename'])):
        path_list.append((os.path.join(input_dir, mat['filename'][i]), mat['frame'][i], mat['crop'][i]))
    return path_list


class VideoSortMatTrainDataSet(VideoTrainDataSet):  # video_train_loader.VideoTrainDataSetを継承

    def __init__(self, pre_processing: transforms.Compose = None, frame_num=4, path_list: list = None,
                 random_crop_size=224, frame_interval=1):
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
        self.shuffle_list = list(range(len(self.data_list[0][1])))

    # イテレートするときに実行されるメソッド．ここをオーバーライドする必要がある．
    def __getitem__(self, index: int) -> tuple:

        frame_list, video_len = self.getitem_init(index)
        assert self.crop_video_len < video_len

        frame_indices = self.data_list[index][1]

        shuffle_frame_indices = list(range(len(frame_indices)))
        random.shuffle(self.shuffle_list)
        for i, v in enumerate(self.shuffle_list):
            shuffle_frame_indices[i] = frame_indices[v] - 1

        pre_processing = lambda image_path: self.pre_processing(Image.open(image_path).convert('RGB'))
        video_tensor = [pre_processing(frame_list[int(i)]) for i in shuffle_frame_indices]
        video_tensor = torch.stack(video_tensor)  # 3次元Tensorを含んだList -> 4次元Tensorに変換

        return video_tensor, torch.tensor(self.shuffle_list)  # 入力画像とそのラベルをタプルとして返す

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
        VideoSortMatTrainDataSet(
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
