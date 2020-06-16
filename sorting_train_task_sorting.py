import argparse
import json
import os
from time import time

import torch
from torch.utils.data import DataLoader

from models.CNN_LSTM_Model import CNN_LSTM
from data_loader.video_sort_train_loader import VideoSortTrainDataSet
from data_loader.video_sort_test_loader import VideoSortTestDataSet
from data_loader.video_test_loader import ucf101_test_path_load
from data_loader.video_train_loader import ucf101_train_path_load

# コマンドライン引数を処理
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--epoch_num', type=int, default=100, required=False)
parser.add_argument('--batch_size', type=int, default=4, required=False)
parser.add_argument('--frame_num', type=int, default=4, required=False)
parser.add_argument('--use_cuda', action='store_true')
parser.add_argument('--use_pretrained_model', action='store_true')
parser.add_argument('--use_bidirectional', action='store_true')
parser.add_argument('--learning_rate', type=float, default=0.01, required=False)
parser.add_argument('--model_save_path', type=str, required=False)
parser.add_argument('--model_load_path', type=str, required=False)
# parser.add_argument('--depth', type=int, default=2, required=False)
parser.add_argument('--model_save_interval', type=int, default=50, required=False)
parser.add_argument('--train_label_path', type=str, required=True)
parser.add_argument('--test_label_path', type=str, required=True)
parser.add_argument('--class_path', type=str, required=True)
parser.add_argument('--no_reset_log_file', action='store_true')
parser.add_argument('--load_epoch_num', action='store_true')
parser.add_argument('--interval_frames', type=int, default=4, required=False)

args = parser.parse_args()
batch_size = args.batch_size
frame_num = args.frame_num
log_train_path = os.path.join(args.output_dir, 'log_train.csv')
log_test_path = os.path.join(args.output_dir, 'log_test.csv')
os.makedirs(args.output_dir, exist_ok=True)
if not args.model_save_path:
    args.model_save_path = os.path.join(args.output_dir, 'model.pth')
json.dump(vars(args), open(os.path.join(args.output_dir, 'args.json'), mode='w'),
          ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))

# データセットを読み込む
train_loader = DataLoader(
    VideoSortTrainDataSet(
        frame_num=frame_num,
        path_load=ucf101_train_path_load(args.dataset_path, args.train_label_path),
        frame_interval=args.interval_frames),
    batch_size=batch_size, shuffle=True)
test_loader = DataLoader(
    VideoSortTestDataSet(
        frame_num=frame_num,
        path_load=ucf101_test_path_load(args.dataset_path, args.test_label_path, args.class_path),
        interval_frame=args.interval_frames),
    batch_size=batch_size, shuffle=False)
train_iterate_len = len(train_loader)
test_iterate_len = len(test_loader)

# 初期設定
# resnet18を取得
Net = CNN_LSTM(args.frame_num, pretrained=args.use_pretrained_model, bidirectional=args.use_bidirectional,
               task='sorting')
criterion = torch.nn.CrossEntropyLoss()  # Loss関数を定義
optimizer = torch.optim.Adam(Net.parameters(), lr=args.learning_rate)  # 重み更新方法を定義
current_epoch = 0

# ログファイルの生成
if not args.no_reset_log_file:
    with open(log_train_path, mode='w') as f:
        f.write('epoch,loss,full_fit_accuracy,per_fit_accuracy,time,learning_rate\n')
    with open(log_test_path, mode='w') as f:
        f.write('epoch,loss,full_fit_accuracy,per_fit_accuracy,time,learning_rate\n')

# CUDA環境の有無で処理を変更
if args.use_cuda:
    criterion = criterion.cuda()
    Net = torch.nn.DataParallel(Net.cuda())
    device = 'cuda'
else:
    device = 'cpu'

# モデルの読み込み
if args.model_load_path:
    checkpoint = torch.load(args.model_load_path)
    Net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    optimizer.lr = args.learning_rate
    if args.load_epoch_num:
        current_epoch = checkpoint['epoch']
    print('complete load model')


def inner_product_loss(outputs: torch.Tensor) -> torch.Tensor:
    # seq_len = outputs.size()[0]  # batch_first = False
    seq_len = outputs.size()[1]  # batch_first = True
    # outputs_ = outputs.permute(1, 0, 2)  # batch_first = False
    outputs_ = outputs  # batch_first = True
    out = torch.zeros(outputs_.size()[0], seq_len * (seq_len - 1) // 2)
    out = out.to(device, non_blocking=True)
    for i in range(outputs_.size()[0]):  # (batch_size, seq_len, class_num)
        index = 0
        for j in range(seq_len):
            for k in range(j, seq_len):
                if j == k:
                    continue
                out[i][index] = torch.abs(torch.dot(outputs_[i][j], outputs_[i][k]))
                index += 1
    return torch.mean(out) * 0.1
    # return torch.sum(out)


# VideoDataTrainDataSetの出力はフレームのみ．ラベルの取得はtrain_loader.dataset.shuffle_listを呼び出すこと
# 訓練を行う
def train(inputs):
    # 演算開始. start calculate.
    # train_loader.dataset.update_shuffle_list()
    """
    [2, 3, 0, 1]
    -> [[2, 3, 0, 1],
        [2, 3, 0, 1]]
    """
    # labels = torch.tensor(train_loader.dataset.shuffle_list)
    # labels = labels.expand(inputs.size()[0], frame_num)
    labels = inputs[1]
    labels = labels.to(device, non_blocking=True)
    # outputs = Net(inputs)  # この記述方法で順伝搬が行われる (seq_len, batch_size, class_num)
    outputs = Net(inputs[0])  # この記述方法で順伝搬が行われる (seq_len, batch_size, class_num)
    optimizer.zero_grad()  # 勾配を初期化
    # loss = criterion(outputs.permute(1, 2, 0), labels) + inner_product_loss(outputs)  # Loss値を計算
    # loss = criterion(outputs.permute(1, 2, 0), labels)  # Loss値を計算 batch_first = False
    loss = criterion(outputs, labels)  # Loss値を計算 batch_first = True
    # loss = criterion(outputs, labels) + inner_product_loss(outputs)  # Loss値を計算 batch_first = True
    loss.backward()  # 逆伝搬で勾配を求める
    optimizer.step()  # 重みを更新
    return outputs, loss.item(), labels


# VideoDataTestDataSetの出力は(フレーム，ラベル)である．
# テストを行う
def test(inputs):
    with torch.no_grad():  # 勾配計算が行われないようにする
        # labels = torch.tensor(inputs[1])
        labels = inputs[1]
        labels = labels.to(device, non_blocking=True)
        outputs = Net(inputs[0])  # この記述方法で順伝搬が行われる
        # loss = criterion(outputs.permute(1, 2, 0), labels) + inner_product_loss(outputs)  # Loss値を計算
        # loss = criterion(outputs.permute(1, 2, 0), labels)  # Loss値を計算 batch_first = False
        loss = criterion(outputs, labels)  # Loss値を計算 batch_first = True
        # loss = criterion(outputs, labels) + inner_product_loss(outputs)  # Loss値を計算 batch_first = True
    return outputs, loss.item(), labels


# 推論を行う
def estimate(data_loader: DataLoader, calc_func, subset: str, epoch_num: int, log_file: str, iterate_len: int,
             get_batch_size_func):
    epoch_loss = 0
    epoch_full_fit_accuracy = 0
    epoch_per_fit_accuracy = 0
    start_time = time()

    for i, data in enumerate(data_loader):
        # 前処理
        inputs = data
        temp_batch_size = get_batch_size_func(inputs)  # batch_size=4 data_len=10 最後に2余るのでこれで対応する
        answer = torch.full_like(torch.zeros(temp_batch_size), fill_value=frame_num).cuda()  # accuracyの計算に使う

        # 演算開始. start calculate.
        outputs, loss, labels = calc_func(inputs)

        # 後処理
        # predicted = torch.max(outputs.permute(1, 0, 2), 2)[1]  # batch_first = False
        predicted = torch.max(outputs, 2)[1]  # batch_first = True
        per_fit_accuracy = (predicted == labels).sum().item() / (batch_size * frame_num)
        full_fit_accuracy = ((predicted == labels).sum(1) == answer).sum().item() / temp_batch_size
        epoch_per_fit_accuracy += per_fit_accuracy
        epoch_full_fit_accuracy += full_fit_accuracy
        epoch_loss += loss
        print(f'{subset}: epoch = {epoch_num + 1}, i = [{i}/{iterate_len - 1}], {loss = }, ' +
              f'{full_fit_accuracy = }, {per_fit_accuracy=}')

    loss_avg = epoch_loss / iterate_len
    full_fit_accuracy_avg = epoch_full_fit_accuracy / iterate_len
    per_fit_accuracy_avg = epoch_per_fit_accuracy / iterate_len
    epoch_time = time() - start_time
    learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
    print(f'{subset}: epoch = {epoch_num + 1}, {loss_avg = }, {full_fit_accuracy_avg = }, ' +
          f'{per_fit_accuracy_avg=}, {epoch_time = }, {learning_rate = }')
    with open(log_file, mode='a') as f:
        f.write(f'{epoch_num + 1},{loss_avg},{full_fit_accuracy_avg},{per_fit_accuracy_avg},' +
                f'{epoch_time},{learning_rate}\n')


# 推論を実行
try:
    for epoch in range(current_epoch, args.epoch_num):
        current_epoch = epoch
        Net.train()
        # estimate(train_loader, train, 'train', epoch, log_train_path, train_iterate_len, lambda x: len(x))
        estimate(train_loader, train, 'train', epoch, log_train_path, train_iterate_len, lambda x: len(x[0]))
        Net.eval()
        estimate(test_loader, test, 'test', epoch, log_test_path, test_iterate_len, lambda x: len(x[0]))
except KeyboardInterrupt:  # Ctrl-Cで保存．
    if args.model_save_path:
        torch.save({
            'epoch': current_epoch,
            'model_state_dict': Net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, args.model_save_path)
        print('complete save model')
        exit(0)

if args.model_save_path:
    torch.save({
        'epoch': args.epoch_num,
        'model_state_dict': Net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, args.model_save_path)
    print('complete save model')
