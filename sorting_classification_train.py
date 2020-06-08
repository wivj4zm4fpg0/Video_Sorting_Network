import argparse
import json
import os
from time import time
import math

import torch
from torch.utils.data import DataLoader

from CNN_LSTM_Model import CNN_LSTM
from video_sorting_classification_train_loader import VideoSortingClassificationTrainDataSet
from video_test_loader import ucf101_test_path_load
from video_train_loader import ucf101_train_path_load

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
    VideoSortingClassificationTrainDataSet(
        frame_num=frame_num,
        path_load=ucf101_train_path_load(args.dataset_path, args.train_label_path),
        interval_frame=args.interval_frames),
    batch_size=batch_size, shuffle=True)
test_loader = DataLoader(
    VideoSortingClassificationTrainDataSet(
        frame_num=frame_num,
        path_load=ucf101_test_path_load(args.dataset_path, args.test_label_path, args.class_path),
        interval_frame=args.interval_frames),
    batch_size=batch_size, shuffle=False)
train_iterate_len = len(train_loader)
test_iterate_len = len(test_loader)

# 初期設定
# resnet18を取得
Net = CNN_LSTM(math.factorial(args.frame_num), pretrained=args.use_pretrained_model,
               bidirectional=args.use_bidirectional, task='classification')
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
    if args.load_epoch_num:
        current_epoch = checkpoint['epoch']
    print('complete load model')


# 双方向の有無で出力の取り方を変える
if args.use_bidirectional:
    # reshape_output = lambda x: mean(x, 1)  # シーケンスの平均を取る (batch_size, seq_len, class_num)
    reshape_output = lambda x: torch.mean(x, 0)  # シーケンスの平均を取る (seq_len, batch_size, class_num)
else:
    # reshape_output = lambda x: x[:, -1, :]  # シーケンスの最後を取る (batch_size, seq_len, class_num)
    reshape_output = lambda x: x[-1, :, :]  # シーケンスの最後を取る (seq_len, batch_size, class_num)


# 訓練を行う
def train(inputs, labels):
    # 演算開始. start calculate.
    outputs = Net(inputs)  # この記述方法で順伝搬が行われる
    optimizer.zero_grad()  # 勾配を初期化
    loss = criterion(reshape_output(outputs), labels)  # Loss値を計算
    loss.backward()  # 逆伝搬で勾配を求める
    optimizer.step()  # 重みを更新
    return outputs, loss.item()


# テストを行う
def test(inputs, labels):
    with torch.no_grad():  # 勾配計算が行われないようにする
        outputs = Net(inputs)  # この記述方法で順伝搬が行われる
        loss = criterion(reshape_output(outputs), labels)  # Loss値を計算
    return outputs, loss.item()


# 推論を行う
def estimate(data_loader, calcu, subset: str, epoch_num: int, log_file: str, iterate_len: int):
    epoch_loss = 0
    epoch_accuracy = 0
    start_time = time()

    for i, data in enumerate(data_loader):
        # 前処理
        inputs, labels = data
        labels = labels.to(device, non_blocking=True)

        # 演算開始. start calculate.
        outputs, loss = calcu(inputs, labels)

        # 後処理
        predicted = torch.max(reshape_output(outputs), 1)[1]
        accuracy = (predicted == labels).sum().item() / batch_size
        epoch_accuracy += accuracy
        epoch_loss += loss
        print(f'{subset}: epoch = {epoch_num + 1}, i = [{i}/{iterate_len - 1}], {loss = }, {accuracy = }')

    loss_avg = epoch_loss / iterate_len
    accuracy_avg = epoch_accuracy / iterate_len
    epoch_time = time() - start_time
    learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
    print(f'{subset}: epoch = {epoch_num + 1}, {loss_avg = }, {accuracy_avg = }, {epoch_time = }, {learning_rate = }')
    with open(log_file, mode='a') as f:
        f.write(f'{epoch_num + 1},{loss_avg},{accuracy_avg},{epoch_time},{learning_rate}\n')

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