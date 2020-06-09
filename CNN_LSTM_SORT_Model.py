import itertools
import math

import torch
from torch import nn
from torchvision.models import resnet18
from torchvision.models.resnet import BasicBlock


class CNNLSTMCombination(nn.Module):
    def __init__(self, frame_num: int = 3, bidirectional: bool = True, pretrained: bool = True):
        super().__init__()

        resnet18_modules = [module for module in (resnet18(pretrained=pretrained).modules())][1:-1]
        resnet18_modules_cut = resnet18_modules[0:4]
        resnet18_modules_cut.extend(
            [module for module in resnet18_modules if type(module) == nn.Sequential and type(module[0]) == BasicBlock])
        resnet18_modules_cut.append(resnet18_modules[-1])
        self.resnet18 = nn.Sequential(*resnet18_modules_cut)
        resnet18_last_dim = 512

        lstm_dim = 512
        batch_first = False
        if bidirectional:
            self.lstm = nn.LSTM(resnet18_last_dim, int(lstm_dim / 2), bidirectional=True, num_layers=2,
                                batch_first=batch_first)
        else:
            self.lstm = nn.LSTM(resnet18_last_dim, lstm_dim, bidirectional=False, num_layers=2, batch_first=batch_first)

        self.combination_list = list(itertools.combinations(list(range(frame_num)), 2))
        self.fc = nn.Linear(lstm_dim * len(self.combination_list), math.factorial(frame_num))
        nn.init.kaiming_normal_(self.fc.weight)

    # xの形は(バッチサイズ, RNNへの入力数, チャンネル数, 解像度, 解像度)の5次元配列である必要がある
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        sequence_length = x.shape[1]
        x = x.permute(1, 0, 2, 3, 4)  # (batch_size, seq_len, img) -> (seq_len, batch_size, img)

        # output_shape -> (seq_len, batch_size, data_size), lstm.batch_first -> Flase
        x = torch.stack([torch.flatten(self.resnet18(x[i]), 1) for i in range(sequence_length)])

        concat_fc = []
        for combination in self.combination_list:
            lstm_input = torch.stack([x[combination[0]], x[combination[1]]])
            lstm_output = self.lstm(lstm_input)[0]
            lstm_mean = torch.mean(lstm_output, 0)
            concat_fc.append(lstm_mean)
        x = self.fc(torch.cat(concat_fc, 1))
        return x


if __name__ == '__main__':
    model = CNNLSTMCombination()
    input = torch.randn(4, 3, 3, 90, 90)
    output = model(input)
    print(f'{output.shape=}')
