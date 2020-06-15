import argparse
import os

import cv2


def resize_image(input_dir: str, width: int, height: int, depth: int):
    for dir in os.listdir(input_dir):
        image_path = os.path.join(input_dir, dir)
        if depth > 0:
            resize_image(image_path, width, height, depth - 1)
        else:
            if not '.jpg' in dir:
                continue
            img = cv2.imread(image_path)
            print(f'{image_path=}')
            if img.shape[0] == height and img.shape[1] == width:
                continue
            img = cv2.resize(img, (width, height))
            cv2.imwrite(image_path, img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=int, default=320, required=False)
    parser.add_argument('--height', type=int, default=240, required=False)
    parser.add_argument('--depth', type=int, default=2, required=False)
    parser.add_argument('--input_dir', type=str, required=True)
    args = parser.parse_args()

    resize_image(args.input_dir, args.width, args.height, args.depth)
