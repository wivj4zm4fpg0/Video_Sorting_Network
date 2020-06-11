import itertools
import math

import torch
from torch import nn
from torchvision.models import resnet18
from torchvision.models.resnet import BasicBlock


class OPN(nn.Module):
    def __init__(self, pretrained: bool = True, frame_num=4):
        super().__init__()

        resnet18_modules = [module for module in (resnet18(pretrained=pretrained).modules())][1:-1]
        resnet18_modules_cut = resnet18_modules[0:4]
        resnet18_modules_cut.extend(
            [module for module in resnet18_modules if type(module) == nn.Sequential and type(module[0]) == BasicBlock])
        resnet18_modules_cut.append(resnet18_modules[-1])
        self.resnet18 = nn.Sequential(*resnet18_modules_cut)
        resnet18_last_dim = 512

        self.comb_fc1 = nn.Linear(resnet18_last_dim * 2, 512)
        class_num = math.factorial(frame_num)
        self.combination_list = list(itertools.combinations(list(range(frame_num)), 2))
        self.comb_fc2 = nn.Linear(512 * len(self.combination_list), class_num)
        nn.init.kaiming_normal_(self.comb_fc1.weight)
        nn.init.kaiming_normal_(self.comb_fc2.weight)

    # xの形は(バッチサイズ, RNNへの入力数, チャンネル数, 解像度, 解像度)の5次元配列である必要がある
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence_length = x.shape[1]
        x = x.permute(1, 0, 2, 3, 4)  # (batch_size, seq_len, img) -> (seq_len, batch_size, img)
        x = torch.stack([torch.flatten(self.resnet18(x[i]), 1) for i in range(sequence_length)])
        comb_feat_list = []
        for comb in self.combination_list:
            comb_feat_list.append(self.comb_fc1(torch.cat([x[comb[0]], x[comb[1]]], 1)))
        x = self.comb_fc2(torch.cat(comb_feat_list, 1))
        return x


if __name__ == '__main__':
    model = CNN_LSTM()
    input = torch.randn(2, 4, 3, 256, 256)
    output = model(input)
    print(f'{output.shape=}')
