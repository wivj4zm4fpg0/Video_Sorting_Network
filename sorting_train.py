import argparse
import json
import os

from torch import nn, optim, save, load, mean
from torch.utils.data import DataLoader

from CNN_LSTM_Model import CNN_LSTM
from LSTM_Action_Recognition import train, estimate
from video_sort_train_loader import VideoSortTrainDataSet, recursive_video_path_load

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
parser.add_argument('--depth', type=int, default=2, required=False)
parser.add_argument('--model_save_interval', type=int, default=50, required=False)

args = parser.parse_args()
batch_size = args.batch_size
frame_num = args.frame_num
log_train_path = os.path.join(args.output_dir, 'log_train.csv')
os.makedirs(args.output_dir, exist_ok=True)
if not args.model_save_path:
    args.model_save_path = os.path.join(args.output_dir, 'model.pth')
json.dump(vars(args), open(os.path.join(args.output_dir, 'args.jsons'), mode='w'),
          ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))

# データセットを読み込む
train_loader = DataLoader(
    VideoSortTrainDataSet(frame_num=frame_num, path_load=recursive_video_path_load(args.dataset_path, args.depth)),
    batch_size=batch_size, shuffle=True)
train_iterate_len = len(train_loader)

# 初期設定
# resnet18を取得
Net = CNN_LSTM(args.class_num, pretrained=args.use_pretrained_model, bidirectional=args.use_bidirectional)
criterion = nn.CrossEntropyLoss()  # Loss関数を定義
optimizer = optim.Adam(Net.parameters(), lr=args.learning_rate)  # 重み更新方法を定義
start_epoch = 0
if args.model_load_path:
    checkpoint = load(args.model_load_path)
    start_epoch = checkpoint['epoch']
    Net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# ログファイルの生成
with open(log_train_path, mode='w') as f:
    f.write('epoch,loss,accuracy,time,learning_rate\n')

# CUDA環境の有無で処理を変更
if args.use_cuda:
    criterion = criterion.cuda()
    Net = nn.DataParallel(Net.cuda())
    device = 'cuda'
else:
    device = 'cpu'

# 双方向の有無で出力の取り方を変える
if args.use_bidirectional:
    reshape_output = lambda x: mean(x, 1)  # シーケンスの平均を取る
else:
    reshape_output = lambda x: x[:, -1, :]  # シーケンスの最後を取る

# 推論を実行
for epoch in range(start_epoch, args.epoch_num):
    Net.train()
    estimate(train_loader, train, 'train', start_epoch, log_train_path, train_iterate_len)
    if (epoch + 1) % args.model_save_interval == 0:
        save({
            'epoch': (epoch + 1),
            'model_state_dict': Net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, args.model_save_path)
