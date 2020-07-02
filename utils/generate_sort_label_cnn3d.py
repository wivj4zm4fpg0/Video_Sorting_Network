import argparse
import os
import re

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, required=True)
parser.add_argument('--cnn3d_frame_num', type=int, default=16, required=False)
parser.add_argument('--output_csv', type=str, required=True)
args = parser.parse_args()
input_dir = args.input_dir
clip_num = 3
cnn3d_frame_num = args.cnn3d_frame_num
crop_video_len = cnn3d_frame_num * clip_num
output_csv = args.output_csv
sort_len = 3
labels = [[[0, 1, 2], [0, 1, 2], 0], [[0, 2, 1], [0, 2, 1], 1], [[1, 0, 2], [1, 0, 2], 2],
          [[2, 1, 0], [0, 1, 2], 0], [[1, 2, 0], [0, 2, 1], 1], [[2, 0, 1], [1, 0, 2], 2]]


def l2s(x: list) -> str:
    return re.sub(r', ', '_', str(x)[1:-1])


with open(output_csv, mode='w') as f:
    f.write('file_name,frames1,frames2,frames3,label1,label2\n')

    for class_ in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_)
        for video in os.listdir(class_path):
            print(f'{video = }')
            video_path = os.path.join(class_path, video)
            video_len = len(os.listdir(video_path))
            print(f'{video_len = }')
            subtracted_frame = video_len - crop_video_len
            interval_frame = subtracted_frame // (clip_num + 1)
            frame_indices = list(range(video_len))
            extracted_frames = frame_indices[interval_frame:interval_frame * clip_num + crop_video_len * (
                    clip_num - 1) + 1:interval_frame + crop_video_len]
            extracted_frame_indices = [list(range(v, v + crop_video_len)) for v in extracted_frames]
            sort_list = [extracted_frame_indices[i][cnn3d_frame_num * j:cnn3d_frame_num * (j + 1) - 1] for j in
                         range(sort_len) for i in range(clip_num)]
            for clips in sort_list:
                for order in labels:
                    label_list = []
                    for v in order[0]:
                        label_list.append(clips[v])
                    f.write(f'{class_}/{video},{l2s(label_list[0])},{l2s(label_list[1])},'
                            + f'{l2s(label_list[2])},{l2s(order[1])},{order[2]}\n')
