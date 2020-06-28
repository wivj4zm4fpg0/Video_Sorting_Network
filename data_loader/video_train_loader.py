import os
from random import randint

import torch
from PIL import Image
from natsort import natsorted
from torch.utils.data import Dataset
from torchvision import transforms

from data_loader.video_transforms import VideoRandomCrop


class VideoTrainDataSet(Dataset):  # torch.utils.data.Datasetを継承

    def __init__(self, pre_processing: transforms.Compose = None, frame_num=4, path_list: list = None,
                 random_crop_size=224, frame_interval=0):

        self.crop_video_len = (frame_num - 1) * frame_interval + frame_num
        self.frame_num = frame_num
        self.data_list = path_list
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

    def getitem_init(self, index: int) -> tuple:

        nat_list = natsorted(os.listdir(self.data_list[index][0]))
        frame_list = [os.path.join(self.data_list[index][0], frame) for frame in nat_list]
        frame_list = [frame for frame in frame_list if '.jpg' in frame or '.png' in frame]
        video_len = len(frame_list)

        # self.pre_processing.transforms[0].set_degree()  # RandomRotationの回転角度を設定
        # RandomCropの設定を行う. 引数に画像サイズが必要なので最初のフレームを渡す
        self.pre_processing.transforms[0].set_param(Image.open(frame_list[0]))
        # RandomHorizontalFlipのソースコード参照．pの値を設定．0なら反転しない，1なら反転する
        self.pre_processing.transforms[1].p = randint(0, 1)

        return frame_list, video_len

    # イテレートするときに実行されるメソッド．ここをオーバーライドする必要がある．
    def __getitem__(self, index: int) -> tuple:

        frame_list, video_len = self.getitem_init(index)
        assert self.crop_video_len < video_len

        start_index = randint(0, video_len - self.crop_video_len)
        frame_indices = range(start_index, start_index + self.crop_video_len, self.frame_interval + 1)

        pre_processing = lambda image_path: self.pre_processing(Image.open(image_path).convert('RGB'))
        video_tensor = [pre_processing(frame_list[i]) for i in frame_indices]
        video_tensor = torch.stack(video_tensor)  # 3次元Tensorを含んだList -> 4次元Tensorに変換
        label = torch.tensor(self.data_list[index][1])
        return video_tensor, label  # 入力画像とそのラベルをタプルとして返す

    def __len__(self) -> int:  # データセットの数を返すようにする
        return len(self.data_list)


if __name__ == '__main__':  # UCF101データセットの読み込みテストを行う

    import argparse
    from data_loader.path_list_loader import recursive_video_path_load
    import numpy as np
    import cv2
    from torchvision.utils import make_grid
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8, required=False)
    parser.add_argument('--frame_interval', type=int, default=0, required=False)

    args = parser.parse_args()

    data_loader = DataLoader(
        VideoTrainDataSet(
            path_list=recursive_video_path_load(args.dataset_path, depth=1),
            random_crop_size=180,
            frame_interval=args.frame_interval
        ),
        batch_size=args.batch_size, shuffle=False
    )


    def image_show(img):  # 画像を表示
        np_img = np.transpose(make_grid(img).numpy(), (1, 2, 0))
        cv2.imshow('image', cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))
        cv2.moveWindow('image', 30, 30)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            exit(0)


    for data in data_loader:
        inputs, labels = data
        print(labels)
        for images_per_batch in inputs:
            image_show(images_per_batch)
