import itertools
import math

import torch
from torch import nn

from models.resnet_3d import generate_model


class cnn3d_OPN(nn.Module):
    def __init__(self, pretrained: bool = True, frame_num=4, resnet_3d_pretrained_model='r3d18_KM_200ep.pth'):
        super().__init__()

        resnet18_3d = generate_model(18)
        if pretrained:
            checkpoint = torch.load(resnet_3d_pretrained_model)
            resnet18_3d.fc = nn.Linear(512, 1039)
            resnet18_3d.load_state_dict(checkpoint['state_dict'])

        self.resnet18_3d = resnet18_3d
        resnet18_3d_last_dim = 512
        cnn_last_dim = 1024
        self.cnn_last = nn.Sequential(
            nn.Linear(resnet18_3d_last_dim, cnn_last_dim),
            nn.BatchNorm1d(cnn_last_dim),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )

        comb_fc1_out = 512
        self.comb_fc1 = nn.Sequential(
            nn.Linear(cnn_last_dim * 2, comb_fc1_out),
            # nn.Linear(resnet18_last_dim * 2, comb_fc1_out),
            nn.BatchNorm1d(comb_fc1_out),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )
        class_num = math.factorial(frame_num) // 2
        self.combination_list = list(itertools.combinations(list(range(frame_num)), 2))
        self.comb_fc2 = nn.Linear(512 * len(self.combination_list), class_num)
        nn.init.kaiming_normal_(self.cnn_last[0].weight)
        nn.init.kaiming_normal_(self.comb_fc1[0].weight)
        nn.init.kaiming_normal_(self.comb_fc2.weight)

    # xの形は(バッチサイズ, RNNへの入力数, チャンネル数, 解像度, 解像度)の5次元配列である必要がある
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence_length = x.shape[1]

        # (batch_size, seq_len, temporal, ch, height, width) -> (seq_len, batch_size, ch, temporal, height, width)
        x = x.permute(1, 0, 3, 2, 4, 5)

        x = torch.stack([self.cnn_last(torch.flatten(self.resnet18_3d(x[i]), 1)) for i in range(sequence_length)])
        # x = torch.stack([torch.flatten(self.resnet18_3d(x[i]), 1) for i in range(sequence_length)])
        comb_feat_list = []
        for comb in self.combination_list:
            comb_feat_list.append(self.comb_fc1(torch.cat([x[comb[0]], x[comb[1]]], 1)))
        x = self.comb_fc2(torch.cat(comb_feat_list, 1))
        return x


if __name__ == '__main__':
    model = cnn3d_OPN(pretrained=False)
    input = torch.randn(2, 4, 16, 3, 96, 96)
    output = model(input)
    print(f'{output.shape=}')
