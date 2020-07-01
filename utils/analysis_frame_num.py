import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir')
parser.add_argument('--frame_num', default=48)
args = parser.parse_args()
input_dir = args.input_dir
frame_num = args.frame_num

result = {}
for class_ in os.listdir(input_dir):
    class_path = os.path.join(input_dir, class_)
    for video in os.listdir(class_path):
        video_path = os.path.join(class_path, video)
        video_len = len(os.listdir(video_path))
        count = video_len // frame_num
        if count in result.keys():
            result[count] += 1
        else:
            result[count] = 0

print(f'{result = }')
