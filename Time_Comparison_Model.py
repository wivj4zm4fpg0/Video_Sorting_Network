import torch
from torch import nn
from torchvision.models import resnet18
from torchvision.models.resnet import BasicBlock


class TimeComparison(nn.Module):
    def __init__(self, bidirectional: bool = True, pretrained: bool = True):
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
            self.lstm_forward = self.bidirectional_forward
            self.lstm = nn.LSTM(resnet18_last_dim, int(lstm_dim / 2), bidirectional=True, num_layers=2,
                                batch_first=batch_first)
        else:
            self.lstm_forward = self.normal_forward
            self.lstm = nn.LSTM(resnet18_last_dim, lstm_dim, bidirectional=False, num_layers=2, batch_first=batch_first)

        # self.fc = nn.Linear(lstm_dim * 2, 2)
        # nn.init.kaiming_normal_(self.fc.weight)

        self.fc1 = nn.Linear(lstm_dim * 2, 4096)
        self.fc2 = nn.Linear(4096, 2)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)

    # xの形は(バッチサイズ, RNNへの入力数, チャンネル数, 解像度, 解像度)の5次元配列である必要がある
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lstm_forward(x)
        # x = self.fc(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def bidirectional_forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        seq_half = seq_len // 2
        x = x.permute(1, 0, 2, 3, 4)  # (batch_size, seq_len, img) -> (seq_len, batch_size, img)

        x0 = torch.stack([torch.flatten(self.resnet18(x[i]), 1) for i in range(0, seq_half)])
        x0 = torch.mean(self.lstm(x0)[0], 0)
        x1 = torch.stack([torch.flatten(self.resnet18(x[i]), 1) for i in range(seq_half, seq_len)])
        x1 = torch.mean(self.lstm(x1)[0], 0)
        x = torch.cat([x0, x1], 1)

        return x

    def normal_forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        seq_half = seq_len // 2
        x = x.permute(1, 0, 2, 3, 4)  # (batch_size, seq_len, img) -> (seq_len, batch_size, img)

        x0 = torch.stack([torch.flatten(self.resnet18(x[i]), 1) for i in range(0, seq_half)])
        x0 = self.lstm(x0)[0][-1]
        x1 = torch.stack([torch.flatten(self.resnet18(x[i]), 1) for i in range(seq_half, seq_len)])
        x1 = self.lstm(x1)[0][-1]
        x = torch.cat([x0, x1], 1)

        return x


if __name__ == '__main__':
    model = TimeComparison(bidirectional=False)
    input = torch.randn(2, 4, 3, 90, 90)
    output = model(input)
    print(f'{output.shape=}')
