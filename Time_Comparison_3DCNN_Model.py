import torch
from torch import nn

from resnet_3d import generate_model


class TimeComparison3DCNN(nn.Module):
    def __init__(self, pretrained: bool = True, resnet_3d_pretrained_model='r3d18_KM_200ep.pth'):
        super().__init__()

        resnet18_3d = generate_model(18)
        if pretrained:
            checkpoint = torch.load(resnet_3d_pretrained_model)
            resnet18_3d.fc = nn.Linear(512, 1039)
            resnet18_3d.load_state_dict(checkpoint['state_dict'])
        self.resnet18_3d = resnet18_3d
        resnet18_3d_last_dim = 512
        self.fc = nn.Linear(resnet18_3d_last_dim * 2, 2)
        nn.init.kaiming_normal_(self.fc.weight)

    # xの形は(バッチサイズ, RNNへの入力数, チャンネル数, 解像度, 解像度)の5次元配列である必要がある
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        seq_half = seq_len // 2
        x0 = x[:, 0:seq_half]
        x1 = x[:, seq_half:seq_len]
        x0 = torch.flatten(self.resnet18_3d(x0.permute(0, 2, 1, 3, 4)), 1)
        x1 = torch.flatten(self.resnet18_3d(x1.permute(0, 2, 1, 3, 4)), 1)
        x = self.fc(torch.cat([x0, x1], 1))
        return x


if __name__ == '__main__':
    model = TimeComparison3DCNN()
    input = torch.randn(2, 32, 3, 90, 90)
    output = model(input)
    print(f'{output.shape=}')
