import argparse
import itertools
import os
import random

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, required=True)
parser.add_argument('--frame_num', type=int, default=4, required=False)
parser.add_argument('--frame_interval', type=int, default=0, required=False)
parser.add_argument('--output_txt', type=str, required=True)
args = parser.parse_args()
input_dir = args.input_dir
frame_num = args.frame_num
frame_interval = args.frame_interval
crop_video_len = (frame_num - 1) * frame_interval + frame_num
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
    column += f', label{i + 1}'
with open(output_txt, mode='w') as f:
    f.write(f'{column}\n')

for class_ in os.listdir(input_dir):
    class_path = os.path.join(input_dir, class_)
    for video in os.listdir(class_path):
        print(f'{video = }')
        video_frame_num = len(os.listdir(os.path.join(class_path, video)))
        assert crop_video_len < video_frame_num
        start_index = random.randint(0, video_frame_num - crop_video_len)
        frame_indices = range(start_index, start_index + crop_video_len, frame_interval + 1)
        label = shuffle_list[random.randint(0, shuffle_len - 1)]
        shuffle_label = list(range(frame_num))
        for i, v in enumerate(label):
            shuffle_label[i] = frame_indices[v]
        with open(output_txt, mode='a') as f:
            f.write(f'{video}, {str(shuffle_label)[1:-1]}\n')
