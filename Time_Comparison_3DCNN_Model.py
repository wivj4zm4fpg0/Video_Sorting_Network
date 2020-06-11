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
        # self.fc = nn.Linear(resnet18_3d_last_dim * 2, 2)
        # nn.init.kaiming_normal_(self.fc.weight)
        self.fc1 = nn.Linear(resnet18_3d_last_dim * 2, 4096)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(4096, 4096)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(4096, 2)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)

    # xの形は(バッチサイズ, RNNへの入力数, チャンネル数, 解像度, 解像度)の5次元配列である必要がある
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        seq_half = seq_len // 2
        x0 = x[:, 0:seq_half]
        x1 = x[:, seq_half:seq_len]
        x0 = torch.flatten(self.resnet18_3d(x0.permute(0, 2, 1, 3, 4)), 1)
        x1 = torch.flatten(self.resnet18_3d(x1.permute(0, 2, 1, 3, 4)), 1)
        # x = self.fc(torch.cat([x0, x1], 1))
        x = self.fc1(torch.cat([x0, x1], 1))
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    model = TimeComparison3DCNN()
    input = torch.randn(2, 32, 3, 90, 90)
    output = model(input)
    print(f'{output.shape=}')
