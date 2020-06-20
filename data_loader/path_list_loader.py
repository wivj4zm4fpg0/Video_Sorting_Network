import os


def ucf101_train_path_load(video_path: str, label_path: str) -> list:
    data_list = []
    with open(label_path) as f:
        label_path_list = [s.strip() for s in f.readlines()]
        for label in label_path_list:
            split_label = label.split(' ')
            data_list.append((os.path.join(video_path, split_label[0][:-4]), int(split_label[1]) - 1))
    return data_list


def ucf101_test_path_load(video_path: str, label_path: str, class_path: str) -> list:
    data_list = []
    class_dict = {}
    with open(class_path) as f:
        class_list = [s.strip() for s in f.readlines()]
        for txt_line in class_list:
            txt_line_split = txt_line.split(' ')
            class_dict[txt_line_split[1]] = int(txt_line_split[0]) - 1
    with open(label_path) as f:
        label_path_list = [s.strip() for s in f.readlines()]
        for label in label_path_list:
            data_list.append((os.path.join(video_path, label[:-4]), class_dict[os.path.split(label)[0]]))
    return data_list


def hmdb51_subset_path_load(input_dir: str, subset_path: str, subset: str) -> list:
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


def recursive_video_path_load(input_dir: str, depth: int = 2, data_list=None):
    if data_list is None:
        data_list = []
    for file_name in os.listdir(input_dir):
        file_name_path = os.path.join(input_dir, file_name)
        if os.path.isfile(file_name_path):
            continue
        if depth > 0:
            recursive_video_path_load(file_name_path, depth - 1, data_list)
        else:
            data_list.append((file_name_path, 0))  # 0はダミー
    return data_list


def generate_path_list(args, subset: str) -> list:
    if args.dataset == 'ucf101':
        if subset == 'train':
            return ucf101_train_path_load(args.dataset_path, args.ucf101_train_label_path)
        elif subset == 'test':
            return ucf101_test_path_load(args.dataset_path, args.ucf101_test_label_path, args.ucf101_class_path)
    if args.dataset == 'hmdb51':
        if subset == 'train':
            return hmdb51_subset_path_load(args.dataset_path, args.hmdb51_subset_path, 'train')
        elif subset == 'test':
            return hmdb51_subset_path_load(args.dataset_path, args.hmdb51_subset_path, 'test')
