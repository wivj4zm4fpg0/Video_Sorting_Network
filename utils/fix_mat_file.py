import argparse
import os
import scipy.io

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir')
parser.add_argument('--input_mat')
args = parser.parse_args()
input_dir = args.input_dir
input_mat = args.input_mat
mat = scipy.io.loadmat(input_mat, squeeze_me=True)

for filename in mat['filename']:
    path = os.path.join(input_dir, filename)
    basename = os.path.basename(filename)
    for i, v in enumerate(os.listdir(os.path.join(input_dir, 'RGB'))):
        if basename.lower() == v.lower() and basename != v:
            command = f'mv {os.path.join(input_dir, "RGB", v)}, {os.path.join(path)}'
            print(command)
            os.system(command)
