import os

import torch
from PIL import Image
from natsort import natsorted
from torch.utils.data import Dataset
from torchvision import transforms


class VideoTestDataSet(Dataset):  # torch.utils.data.Datasetを継承

    def __init__(self, pre_processing: transforms.Compose = None, frame_num=4, path_list: list = None,
                 center_crop_size=224, frame_interval=0):

        self.crop_video_len = (frame_num - 1) * frame_interval + frame_num
        self.pre_frame_num = self.crop_video_len // 2
        if self.crop_video_len % 2 == 0:
            self.post_frame_num = self.pre_frame_num
        else:
            self.post_frame_num = self.pre_frame_num + 1
        self.frame_interval = frame_interval
        self.data_list = path_list

        if pre_processing:
            self.pre_processing = pre_processing
        else:
            self.pre_processing = transforms.Compose([
                transforms.CenterCrop(center_crop_size),
                transforms.ToTensor(),  # Tensor型へ変換
                transforms.Normalize((0, 0, 0), (1, 1, 1))  # 画素値が0と1の間になるように正規化
            ])

    # イテレートするときに実行されるメソッド．ここをオーバーライドする必要がある．
    def __getitem__(self, index: int) -> tuple:
        #  真ん中のフレームを抽出する
        frame_list = natsorted(os.listdir(self.data_list[index][0]))
        video_len = len(frame_list)
        assert self.crop_video_len < video_len
        video_medium_len = int(video_len / 2)
        frame_indices = list(range(video_medium_len - self.pre_frame_num, video_medium_len + self.post_frame_num,
                                   self.frame_interval + 1))
        pre_processing = lambda image_path: self.pre_processing(Image.open(image_path).convert('RGB'))
        video_tensor = [pre_processing(os.path.join(self.data_list[index][0], frame_list[i])) for i in frame_indices]
        video_tensor = torch.stack(video_tensor)  # 3次元Tensorを含んだList -> 4次元Tensorに変換
        label = self.data_list[index][1]
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
    parser.add_argument('--batch_size', type=int, default=1, required=False)

    args = parser.parse_args()

    data_loader = DataLoader(
        VideoTestDataSet(
            path_list=recursive_video_path_load(args.dataset_path, depth=1),
            frame_num=4,
            frame_interval=1
        ),
        batch_size=args.batch_size, shuffle=True
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
