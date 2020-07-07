import argparse
import json
import os
from time import time

import torch
from torch.utils.data import DataLoader

from data_loader.video_sort_label_train_loader import VideoCNN3dSortLabelTrainDataSet, load_sort_label
from models.CNN3D_RNN_Model import CNN3dRNN

# コマンドライン引数を処理
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, required=True)
parser.add_argument('--input_csv', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--epoch_num', type=int, default=50, required=False)
parser.add_argument('--batch_size', type=int, default=4, required=False)
parser.add_argument('--use_cuda', action='store_true')
parser.add_argument('--use_pretrained_model', action='store_true')
parser.add_argument('--use_bidirectional', action='store_true')
parser.add_argument('--learning_rate', type=float, default=0.01, required=False)
parser.add_argument('--model_load_path', type=str, required=False)
parser.add_argument('--no_reset_log_file', action='store_true')
parser.add_argument('--load_epoch_num', action='store_true')
parser.add_argument('--cnn3d_frame_num', type=int, default=16, required=False)

args = parser.parse_args()
batch_size = args.batch_size
cnn3d_frame_num = args.cnn3d_frame_num
log_train_path = os.path.join(args.output_dir, 'log_train.csv')
model_save_path = os.path.join(args.output_dir, 'model.pth')
os.makedirs(args.output_dir, exist_ok=True)
json.dump(vars(args), open(os.path.join(args.output_dir, 'args.json'), mode='w'),
          ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))

# データセットを読み込む
train_loader = DataLoader(
    VideoCNN3dSortLabelTrainDataSet(
        path_list=load_sort_label(args.input_dir, args.input_csv)),
    batch_size=batch_size, shuffle=True)
train_iterate_len = len(train_loader)

# 初期設定
# resnet18を取得
frame_num = 3
Net = CNN3dRNN(class_num=frame_num, pretrained=args.use_pretrained_model, bidirectional=args.use_bidirectional)
criterion = torch.nn.CrossEntropyLoss()  # Loss関数を定義
optimizer = torch.optim.Adam(Net.parameters(), lr=args.learning_rate)  # 重み更新方法を定義
current_epoch = 0

# ログファイルの生成
if not args.no_reset_log_file:
    with open(log_train_path, mode='w') as f:
        f.write('epoch,loss,accuracy,time,learning_rate\n')

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
    seq_len = outputs.size()[0]  # batch_first = False
    outputs_ = outputs.permute(1, 0, 2)  # batch_first = False
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
def train(inputs, labels):
    # 演算開始. start calculate.
    outputs = Net(inputs)  # この記述方法で順伝搬が行われる (seq_len, batch_size, class_num)
    optimizer.zero_grad()  # 勾配を初期化
    # loss = criterion(outputs.permute(1, 2, 0), labels) + inner_product_loss(outputs)  # Loss値を計算 batch_first = False
    loss = criterion(torch.mean(outputs, 0), labels)  # Loss値を計算 batch_first = False
    loss.backward()  # 逆伝搬で勾配を求める
    optimizer.step()  # 重みを更新
    return outputs, loss.item()


# 推論を行う
def estimate(data_loader: DataLoader, calc_func, subset: str, epoch_num: int, log_file: str, iterate_len: int):
    epoch_loss = 0
    epoch_accuracy = 0
    start_time = time()

    for i, data in enumerate(data_loader):
        # 前処理
        inputs, labels = data
        labels = labels.to(device, non_blocking=True)
        # 演算開始. start calculate.
        outputs, loss = calc_func(inputs, labels)

        # 後処理
        predicted = torch.max(torch.mean(outputs, 0), 1)[1]  # batch_first = False
        accuracy = (predicted == labels).sum().item() / len(inputs)
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
        estimate(train_loader, train, 'train', epoch, log_train_path, train_iterate_len)
except KeyboardInterrupt:  # Ctrl-Cで保存．
    pass

torch.save({
    'epoch': current_epoch,
    'model_state_dict': Net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, model_save_path)
print('complete save model')
