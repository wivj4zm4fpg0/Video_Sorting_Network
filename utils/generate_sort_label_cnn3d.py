import argparse
import itertools
import os
import random
import re

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, required=True)
parser.add_argument('--frame_num', type=int, default=4, required=False)
parser.add_argument('--cnn3d_frame_num', type=int, default=8, required=False)
parser.add_argument('--output_txt', type=str, required=True)
args = parser.parse_args()
input_dir = args.input_dir
frame_num = args.frame_num
cnn3d_frame_num = args.cnn3d_frame_num
crop_video_len = cnn3d_frame_num * frame_num
output_txt = args.output_txt

sort_seq = list(itertools.permutations(list(range(frame_num)), frame_num))
shuffle_list = []
for v in sort_seq:
    v = list(v)
    if v[::-1] in shuffle_list:
        continue
    shuffle_list.append(v)
shuffle_len = len(shuffle_list)

column = 'name'
for i in range(frame_num):
    column += f',label{i + 1}'
with open(output_txt, mode='w') as f:
    f.write(f'{column},order\n')

count = 0
for class_ in os.listdir(input_dir):
    class_path = os.path.join(input_dir, class_)
    for video in os.listdir(class_path):
        print(f'{video = }')
        video_frame_num = len(os.listdir(os.path.join(class_path, video)))
        if crop_video_len <= video_frame_num:
            print(f'{count = }')
            count += 1
            continue
        start_index = random.randint(0, video_frame_num - crop_video_len)
        frame_indices = list(range(video_frame_num))[start_index:start_index + crop_video_len:cnn3d_frame_num]
        frame_indices = [list(range(frame_index, frame_index + cnn3d_frame_num + 1)) for frame_index in frame_indices]
        label = shuffle_list[random.randint(0, shuffle_len - 1)]
        shuffle_label = list(range(frame_num))
        for i, v in enumerate(label):
            shuffle_label[i] = [frame_indices[v][0], frame_indices[v][-1]]
        label_str = ''
        for v in shuffle_label:
            label_str += f'{re.sub(", ", "_", str(v)[1:-1])},'
        order = re.sub(', ', '_', str(label)[1:-1])
        with open(output_txt, mode='a') as f:
            f.write(f'{video},{label_str[:-1]},{order}\n')
