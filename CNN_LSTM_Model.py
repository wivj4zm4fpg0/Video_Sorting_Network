import torch
from torch import nn
from torchvision.models import resnet18
from torchvision.models.resnet import BasicBlock


class CNN_LSTM(nn.Module):
    def __init__(self, class_num: int = 101, bidirectional: bool = True, pretrained: bool = True):
        super().__init__()

        resnet18_modules = [module for module in (resnet18(pretrained=pretrained).modules())][1:-1]
        resnet18_modules_cut = resnet18_modules[0:4]
        resnet18_modules_cut.extend(
            [module for module in resnet18_modules if type(module) == nn.Sequential and type(module[0]) == BasicBlock])
        resnet18_modules_cut.append(resnet18_modules[-1])
        self.resnet18 = nn.Sequential(*resnet18_modules_cut)
        resnet18_last_dim = 512

        lstm_dim = 512
        if bidirectional:
            self.lstm = nn.LSTM(resnet18_last_dim, int(lstm_dim / 2), bidirectional=True, num_layers=2,
                                batch_first=True)
        else:
            self.lstm = nn.LSTM(resnet18_last_dim, lstm_dim, bidirectional=False, num_layers=2, batch_first=True)

        self.fc = nn.Linear(lstm_dim, class_num)
        nn.init.kaiming_normal_(self.fc.weight)

    # xの形は(バッチサイズ, RNNへの入力数, チャンネル数, 解像度, 解像度)の5次元配列である必要がある
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        batch_size = x.shape[0]
        sequence_length = x.shape[1]
        #
        # # (バッチサイズ x RNNへの入力数, チャンネル数, 解像度, 解像度)の4次元配列に変換する
        # x = x.view(batch_size * sequence_length, x.shape[2], x.shape[3], x.shape[4])
        # x = self.resnet18(x)
        # x = x.view(batch_size, sequence_length, -1)

        # 非推奨．CNNが学習されないっぽい．no recommended. It seems that CNN is not learned.
        # x = torch.stack([torch.flatten(self.resnet18(x[i]), 1) for i in range(batch_size)])

        fs = torch.zeros(batch_size, sequence_length, self.lstm_input_size).cuda()
        for i in range(batch_size):
            cnn = self.resnet18(x[i])
            cnn = torch.flatten(cnn, 1)
            cnn = self.fc_pre(cnn)
            fs[i, :, :] = cnn

        x = self.lstm(x)[0]
        x = self.fc(x)
        return x  # (seq_len, batch_size, クラス数}の形になっている


if __name__ == '__main__':
    model = CNN_LSTM()
    input = torch.randn(2, 4, 3, 256, 256)
    output = model(input)
    print(f'{output.shape=}')
