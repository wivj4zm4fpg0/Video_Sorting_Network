import os
from random import randint

import cv2
import numpy as np
import torch
from PIL import Image
from natsort import natsorted
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid

from data_loader.video_transforms import VideoRandomCrop


# データセットの形式に合わせて新しく作る
def ucf101_train_path_load(video_path: str, label_path: str) -> list:
    data_list = []
    with open(label_path) as f:
        label_path_list = [s.strip() for s in f.readlines()]
        for label in label_path_list:
            split_label = label.split(' ')
            data_list.append((os.path.join(video_path, split_label[0][:-4]), int(split_label[1]) - 1))
    return data_list


class VideoTrainDataSet(Dataset):  # torch.utils.data.Datasetを継承

    def __init__(self, pre_processing: transforms.Compose = None, frame_num=4, path_load: list = None,
                 random_crop_size=224, frame_interval=0):

        self.frame_num = frame_num * (frame_interval + 1)
        self.data_list = path_load
        self.frame_interval = frame_interval

        if pre_processing:
            self.pre_processing = pre_processing
        else:
            self.pre_processing = transforms.Compose([
                # VideoRandomRotation(0),
                VideoRandomCrop(random_crop_size),
                transforms.RandomHorizontalFlip(),  # ランダムで左右回転
                transforms.ToTensor(),  # Tensor型へ変換
                transforms.Normalize((0, 0, 0), (1, 1, 1))  # 画素値が0と1の間になるように正規化
            ])

    # イテレートするときに実行されるメソッド．ここをオーバーライドする必要がある．
    def __getitem__(self, index: int) -> tuple:
        frame_list = \
            [os.path.join(self.data_list[index][0], frame) for frame in natsorted(os.listdir(self.data_list[index][0]))]
        frame_list = [frame for frame in frame_list if '.jpg' in frame or '.png' in frame]
        video_len = len(frame_list)
        # {frame_index + 0, frame_index + 1, ..., frame_index + self.frame_num - 1}番号のフレームを取得するのに使う
        frame_index = randint(0, video_len - self.frame_num - 1)
        frame_indices = range(frame_index, frame_index + self.frame_num, self.frame_interval + 1)

        # self.pre_processing.transforms[0].set_degree()  # RandomRotationの回転角度を設定
        # RandomCropの設定を行う. 引数に画像サイズが必要なので最初のフレームを渡す
        self.pre_processing.transforms[0].set_param(Image.open(frame_list[0]))
        # RandomHorizontalFlipのソースコード参照．pの値を設定．0なら反転しない，1なら反転する
        self.pre_processing.transforms[1].p = randint(0, 1)

        pre_processing = lambda image_path: self.pre_processing(Image.open(image_path).convert('RGB'))
        # リスト内包表記で検索
        video_tensor = [pre_processing(frame_list[i]) for i in frame_indices]
        video_tensor = torch.stack(video_tensor)  # 3次元Tensorを含んだList -> 4次元Tensorに変換
        label = self.data_list[index][1]
        return video_tensor, label  # 入力画像とそのラベルをタプルとして返す

    def __len__(self) -> int:  # データセットの数を返すようにする
        return len(self.data_list)


if __name__ == '__main__':  # UCF101データセットの読み込みテストを行う

    import argparse
    from data_loader.video_sort_train_loader import recursive_video_path_load

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8, required=False)
    parser.add_argument('--frame_interval', type=int, default=0, required=False)

    args = parser.parse_args()

    data_loader = DataLoader(
        VideoTrainDataSet(
            path_load=recursive_video_path_load(args.dataset_path, depth=1),
            random_crop_size=180,
            frame_interval=args.frame_interval
        ),
        batch_size=args.batch_size, shuffle=False
    )


    def image_show(img):  # 画像を表示
        np_img = np.transpose(make_grid(img).numpy(), (1, 2, 0))
        cv2.imshow('image', cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))
        cv2.moveWindow('image', 100, 200)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            exit(0)


    for input_images, input_label in data_loader:
        print(input_label)
        for images_per_batch in input_images:
            image_show(images_per_batch)
