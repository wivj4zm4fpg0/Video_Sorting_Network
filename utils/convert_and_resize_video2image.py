import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir')
parser.add_argument('--output_dir')
parser.add_argument('--width', default=160)
parser.add_argument('--height', default=120)
args = parser.parse_args()
input_dir = args.input_dir
output_dir = args.output_dir
width = args.width
height = args.height

for class_ in os.listdir(input_dir):
    class_path = os.path.join(input_dir, class_)
    for video in os.listdir(class_path):
        video_path = os.path.join(class_path, video)
        output_path = os.path.join(output_dir, class_, video)[:-4]
        os.makedirs(output_path, exist_ok=True)
        command = f'ffmpeg -y -i {video_path} -s {width}x{height} {output_path}/image_%05d.jpg'
        print(command)
        os.system(command)
