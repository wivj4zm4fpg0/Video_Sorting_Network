import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid


# データセットの形式に合わせて新しく作る
def ucf101_test_path_load(video_path: str, label_path: str, class_path: str) -> list:
    data_list = []
    class_dict = {}
    with open(class_path) as f:
        class_list = [s.strip() for s in f.readlines()]
        for txt_line in class_list:
            txt_line_split = txt_line.split(' ')
            class_dict[txt_line_split[1]] = int(txt_line_split[0]) - 1
    with open(label_path) as f:
        label_path_list = [s.strip() for s in f.readlines()]
        for label in label_path_list:
            data_list.append((os.path.join(video_path, label[:-4]), class_dict[os.path.split(label)[0]]))
    return data_list


class VideoTestDataSet(Dataset):  # torch.utils.data.Datasetを継承

    def __init__(self, pre_processing: transforms.Compose = None, frame_num=4, path_load: list = None,
                 center_crop_size=224, frame_interval=0):

        self.frame_num = int(frame_num * (frame_interval + 1) / 2)
        self.frame_interval = frame_interval
        self.data_list = path_load

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
        frame_list = os.listdir(self.data_list[index][0])
        video_medium_len = int(len(frame_list) / 2)
        frame_indices = \
            list(range(video_medium_len - self.frame_num, video_medium_len + self.frame_num, self.frame_interval + 1))
        pre_processing = lambda image_path: self.pre_processing(Image.open(image_path).convert('RGB'))
        video_tensor = [pre_processing(os.path.join(self.data_list[index][0], frame_list[i])) for i in frame_indices]
        video_tensor = torch.stack(video_tensor)  # 3次元Tensorを含んだList -> 4次元Tensorに変換
        label = self.data_list[index][1]
        return video_tensor, label  # 入力画像とそのラベルをタプルとして返す

    def __len__(self) -> int:  # データセットの数を返すようにする
        return len(self.data_list)


if __name__ == '__main__':  # UCF101データセットの読み込みテストを行う

    import argparse
    from video_sort_train_loader import recursive_video_path_load

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1, required=False)

    args = parser.parse_args()

    data_loader = DataLoader(
        VideoTestDataSet(
            path_load=recursive_video_path_load(args.dataset_path, depth=1),
            frame_num=4,
            frame_interval=3
        ),
        batch_size=args.batch_size, shuffle=True
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
