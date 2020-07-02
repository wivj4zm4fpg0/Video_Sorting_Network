import argparse
import os
import re

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, required=True)
parser.add_argument('--cnn3d_frame_num', type=int, default=16, required=False)
parser.add_argument('--output_csv', type=str, required=True)
parser.add_argument('--subset_path', type=str, required=True)
args = parser.parse_args()
input_dir = args.input_dir
subset_path = args.subset_path
clip_num = 3
cnn3d_frame_num = args.cnn3d_frame_num
crop_video_len = cnn3d_frame_num * clip_num
output_csv = args.output_csv
sort_len = 3
labels = [[[0, 1, 2], [0, 1, 2], 0], [[0, 2, 1], [0, 2, 1], 1], [[1, 0, 2], [1, 0, 2], 2],
          [[2, 1, 0], [0, 1, 2], 0], [[1, 2, 0], [0, 2, 1], 1], [[2, 0, 1], [1, 0, 2], 2]]


def ucf101_train_path_load(video_path: str, label_path: str) -> list:
    data_list = []
    with open(label_path) as f:
        label_path_list = [s.strip() for s in f.readlines()]
        for label in label_path_list:
            split_label = label.split(' ')[0][:-4]
            data_list.append((os.path.join(video_path, split_label), split_label))
    return data_list


def l2s(x: list) -> str:
    return re.sub(r', ', '_', str(x)[1:-1])


data_list = ucf101_train_path_load(input_dir, subset_path)
with open(output_csv, mode='w') as f:
    f.write('file_name,frames1,frames2,frames3,label1,label2\n')

    for video, video_name in data_list:
        video_len = len(os.listdir(video))
        split_num = video_len // crop_video_len
        if split_num > sort_len:
            split_num = sort_len
        subtracted_frame = video_len - crop_video_len * split_num
        interval_frame = subtracted_frame // (split_num + 1)
        frame_indices = list(range(video_len))
        extracted_frames = frame_indices[interval_frame:interval_frame * split_num + crop_video_len * (
                split_num - 1) + 1:interval_frame + crop_video_len]
        extracted_frame_indices = [list(range(v, v + crop_video_len)) for v in extracted_frames]
        sort_list = []
        for i in range(split_num):
            clip_list = []
            for j in range(clip_num):
                clip_list.append(extracted_frame_indices[i][cnn3d_frame_num * j:cnn3d_frame_num * (j + 1)])
            sort_list.append(clip_list)
        for clips in sort_list:
            for order in labels:
                label_list = []
                for v in order[0]:
                    label_list.append(clips[v])
                f.write(f'{video_name},{l2s(label_list[0])},{l2s(label_list[1])},'
                        + f'{l2s(label_list[2])},{l2s(order[1])},{order[2]}\n')
