import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir')
parser.add_argument('--output_dir')
args = parser.parse_args()
input_dir = args.input_dir
output_dir = args.output_dir

for class_ in os.listdir(input_dir):
    class_path = os.path.join(input_dir, class_)
    for video in os.listdir(class_path):
        video_path = os.path.join(class_path, video)
        command = f'cp {video_path} {output_dir}'
        print(command)
        os.system(command)
