import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, required=True)
parser.add_argument('--output_txt', type=str, required=True)
args = parser.parse_args()

input_dir = args.input_dir
output_txt = args.output_txt

train = 0
test = 0
none = 0
video_name_list = []
for i, txt in enumerate(os.listdir(input_dir)):
    if i % 3 != 0:
        continue
    with open(os.path.join(input_dir, txt), 'r') as f:
        for line in [s.strip().split(' ') for s in f.readlines()]:
            if line[1] == '0':
                none += 1
            elif line[1] == '1':
                train += 1
            elif line[1] == '2':
                test += 1
            else:
                print('no')
            video_name_list.append([line[0], line[1]])

print(f'{train = }')
print(f'{test = }')
print(f'{none = }')
print(f'{train + test + none = }')

none_count = 0
for i, video_name in enumerate(video_name_list):
    if video_name[1] == '0':
        if none_count < (none * 4 / 5):
            video_name[1] = '1'
            none_count += 1
        else:
            video_name[1] = '2'

with open(output_txt, mode='w') as f:
    for video_name, subset in video_name_list:
        f.write(f'{video_name} {subset}\n')

train = 0
test = 0
none = 0
with open(output_txt, mode='r') as f:
    for line in [s.strip().split(' ') for s in f.readlines()]:
        if line[1] == '0':
            none += 1
        elif line[1] == '1':
            train += 1
        elif line[1] == '2':
            test += 1
        else:
            print('no')
print(f'{train = }')
print(f'{test = }')
print(f'{none = }')
print(f'{train + test + none = }')

# video_num = 0
# correct_video_num = 0
# for class_dir in os.listdir(video_dir):
#     if '.rar' in class_dir:
#         continue
#     class_dir_path = os.path.join(video_dir, class_dir)
#     for video in os.listdir(class_dir_path):
#         video_num += 1
#         if video in video_name_list:
#             correct_video_num += 1
# print(f'{video_num = }')
# print(f'{correct_video_num = }')
