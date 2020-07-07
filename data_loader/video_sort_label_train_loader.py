import os

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from data_loader.video_train_loader import VideoTrainDataSet


def load_sort_label(input_dir: str, input_label: str):
    path_list = []
    input_csv = pd.read_csv(input_label, sep=',', engine='python')
    for i in range(len(input_csv)):
        file_name = os.path.join(input_dir, input_csv.loc[i][0])
        frame1 = [int(v) for v in input_csv.loc[i][1].split('_')]
        frame2 = [int(v) for v in input_csv.loc[i][2].split('_')]
        frame3 = [int(v) for v in input_csv.loc[i][3].split('_')]
        label1 = [int(v) for v in input_csv.loc[i][4].split('_')]
        label2 = int(input_csv.loc[i][5])
        path_list.append((file_name, frame1, frame2, frame3, label1, label2))
    return path_list


class VideoCNN3dSortLabelTrainDataSet(VideoTrainDataSet):  # video_train_loader.VideoTrainDataSetを継承

    def __init__(self, path_list: list = None):
        super().__init__(path_list=path_list)

    # イテレートするときに実行されるメソッド．ここをオーバーライドする必要がある．
    def __getitem__(self, index: int) -> tuple:
        frame_list, _ = self.getitem_init(index)

        frame1 = self.data_list[index][1]
        frame2 = self.data_list[index][2]
        frame3 = self.data_list[index][3]
        label = self.data_list[index][5]  # 4-> per frame, 5-> classification

        pre_processing = lambda image_path: self.pre_processing(Image.open(image_path).convert('RGB'))
        output_tensor = []
        for frame_indices in [frame1, frame2, frame3]:
            video_tensor = [pre_processing(frame_list[i]) for i in frame_indices]
            video_tensor = torch.stack(video_tensor)  # 3次元Tensorを含んだList -> 4次元Tensorに変換
            output_tensor.append(video_tensor)
        output_tensor = torch.stack(output_tensor)

        return output_tensor, torch.tensor(label)  # 入力画像とそのラベルをタプルとして返す

    def __len__(self) -> int:  # データセットの数を返すようにする
        return len(self.data_list)


if __name__ == '__main__':  # UCF101データセットの読み込みテストを行う

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--input_csv', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=3, required=False)

    args = parser.parse_args()

    data_loader = DataLoader(
        VideoCNN3dSortLabelTrainDataSet(
            path_list=load_sort_label(args.dataset_path, args.input_csv)),
        batch_size=args.batch_size, shuffle=True
    )


    def image_show(img):  # 画像を表示
        np_img = np.transpose(make_grid(img).numpy(), (1, 2, 0))
        cv2.imshow('image', cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))
        cv2.moveWindow('image', 100, 100)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            exit(0)


    for data in data_loader:
        inputs, labels = data
        print(f'{labels = }')
        for images_per_batch in inputs:
            for images in images_per_batch:
                image_show(images)
