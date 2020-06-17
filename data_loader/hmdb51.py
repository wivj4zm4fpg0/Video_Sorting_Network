import os


# train = 1, test = 2
def hmdb_subset_path_load(input_dir: str, subset_path: str, subset: str) -> list:
    data_list = []
    subset_dict = {}
    subset = {'train': '1', 'test': '2'}[subset]
    with open(subset_path) as f:
        subset_list = [s.strip() for s in f.readlines()]
        for subset_line in subset_list:
            video_name, subset_name = subset_line.split(' ')
            subset_dict[video_name[:-4]] = subset_name
    for class_id, class_ in enumerate(os.listdir(input_dir)):
        class_path = os.path.join(input_dir, class_)
        for video in os.listdir(class_path):
            video_path = os.path.join(class_path, video)
            if subset_dict[video] == subset:
                data_list.append((video_path, class_id))

    return data_list


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--subset_path', type=str, required=True)
    parser.add_argument('--subset', type=str, default='train', required=False)
    args = parser.parse_args()

    print(hmdb_subset_path_load(args.input_dir, args.subset_path, args.subset))
