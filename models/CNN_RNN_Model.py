import torch
from torch import nn
from torchvision.models import resnet18
from torchvision.models.resnet import BasicBlock


class CNN_RNN(nn.Module):
    def __init__(self, class_num: int = 101, bidirectional: bool = True, pretrained: bool = True):
        super().__init__()

        resnet18_modules = [module for module in (resnet18(pretrained=pretrained).modules())][1:-1]
        resnet18_modules_cut = resnet18_modules[0:4]
        resnet18_modules_cut.extend(
            [module for module in resnet18_modules if type(module) == nn.Sequential and type(module[0]) == BasicBlock])
        resnet18_modules_cut.append(resnet18_modules[-1])
        self.cnn = nn.Sequential(*resnet18_modules_cut)
        cnn_last_dim = 512

        rnn_dim = 512
        num_layers = 2
        if bidirectional:
            # self.rnn = nn.LSTM(cnn_last_dim, int(rnn_dim / 2), bidirectional=True, num_layers=num_layers)
            self.rnn = nn.GRU(cnn_last_dim, int(rnn_dim / 2), bidirectional=True, num_layers=num_layers)
        else:
            # self.rnn = nn.LSTM(cnn_last_dim, rnn_dim, bidirectional=False, num_layers=num_layers)
            self.rnn = nn.GRU(cnn_last_dim, rnn_dim, bidirectional=False, num_layers=num_layers)

        self.fc = nn.Linear(rnn_dim, class_num)
        nn.init.kaiming_normal_(self.fc.weight)

    # xの形は(バッチサイズ, RNNへの入力数, チャンネル数, 解像度, 解像度)の5次元配列である必要がある
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence_length = x.shape[1]
        x = x.permute(1, 0, 2, 3, 4)  # (batch_size, seq_len, img) -> (seq_len, batch_size, img)
        x = torch.stack([torch.flatten(self.cnn(x[i]), 1) for i in range(sequence_length)])
        x = self.rnn(x)[0]
        x = self.fc(x)
        return x


if __name__ == '__main__':
    model = CNN_RNN()
    input = torch.randn(2, 4, 3, 256, 256)
    output = model(input)
    print(f'output.shape = {output.shape}')
