import os
import re


def make_path_list(input_dir: str, output_dir: str, depth=1):
    path_list = []
    for input in os.listdir(input_dir):
        input_path = os.path.join(input_dir, input)
        output_path = os.path.join(output_dir, input)
        if depth > 0:
            path_list.extend(make_path_list(input_path, output_path, depth - 1))
        else:
            output_path = re.sub(r'\.(avi|mp4|webm)', '', output_path)
            path_list.append((input_path, output_path))
    return path_list


def extract_frame(input_path: str, output_path: str):
    os.makedirs(output_path, exist_ok=True)
    command = f'ffmpeg -y -i {input_path} {os.path.join(output_path, "image_%05d.jpg")}'
    print(command)
    os.system(command)


if __name__ == '__main__':
    import argparse
    from joblib import Parallel, delayed

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--depth', type=int, default=1, required=False)
    args = parser.parse_args()

    path_list = make_path_list(args.input_dir, args.output_dir, args.depth)
    Parallel(n_jobs=-1)([
        delayed(extract_frame)(
            input_path=path_list[i][0],
            output_path=path_list[i][1]
        ) for i in range(len(path_list))])
