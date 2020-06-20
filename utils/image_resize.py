import argparse
import os

import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--width', type=int, default=320, required=False)
parser.add_argument('--height', type=int, default=240, required=False)
parser.add_argument('--input_dir', type=str, required=True)
args = parser.parse_args()

input_dir = args.input_dir
width = args.width
height = args.height

for class_ in os.listdir(input_dir):
    class_path = os.path.join(input_dir, class_)
    for video in os.listdir(class_path):
        video_path = os.path.join(class_path, video)
        frame_list = os.listdir(video_path)
        first_img = cv2.imread(os.path.join(video_path, frame_list[0]))
        if first_img.shape[0] == height and first_img.shape[1] == width:
            continue
        else:
            for frame in frame_list:
                frame_path = os.path.join(video_path, frame)
                img = cv2.imread(frame_path)
                img = cv2.resize(img, (width, height))
                cv2.imwrite(frame_path, img)
            print(f'resize {video_path}')
