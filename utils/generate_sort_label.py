import argparse
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
shuffle_list = list(range(frame_num))

for class_ in os.listdir(input_dir):
    class_path = os.path.join(input_dir, class_)
    for video in os.listdir(class_path):
        print(f'{video = }')
        video_frame_num = len(os.listdir(os.path.join(class_path, video)))
        assert crop_video_len < video_frame_num
        random.shuffle(shuffle_list)
        start_index = random.randint(0, video_frame_num - crop_video_len)
        label = range(start_index, start_index + crop_video_len, frame_interval + 1)
        shuffle_label = list(range(frame_num))
        for i, v in enumerate(shuffle_list):
            shuffle_label[i] = label[v]
        with open(output_txt, mode='a') as f:
            f.write(f'{video}, {str(shuffle_label)[1:-1]}\n')
