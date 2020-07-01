import torch
from torch import nn

from models.resnet_3d import generate_model


class CNN3dRNN(nn.Module):
    def __init__(self, class_num: int = 101, bidirectional: bool = True, pretrained: bool = True,
                 resnet_3d_pretrained_model='r3d18_KM_200ep.pth'):
        super().__init__()

        resnet18_3d = generate_model(18)
        if pretrained:
            checkpoint = torch.load(resnet_3d_pretrained_model)
            resnet18_3d.fc = nn.Linear(512, 1039)
            resnet18_3d.load_state_dict(checkpoint['state_dict'])

        self.resnet18_3d = resnet18_3d

        resnet18_3d_last_dim = 512

        lstm_dim = 512
        if bidirectional:
            # self.rnn = nn.LSTM(resnet18_3d_last_dim, lstm_dim // 2, bidirectional=True, num_layers=2)
            self.rnn = nn.GRU(resnet18_3d_last_dim, lstm_dim // 2, bidirectional=True, num_layers=2)
        else:
            # self.rnn = nn.LSTM(resnet18_3d_last_dim, lstm_dim, bidirectional=False, num_layers=2)
            self.rnn = nn.GRU(resnet18_3d_last_dim, lstm_dim, bidirectional=False, num_layers=2)

        self.fc = nn.Linear(lstm_dim, class_num)
        nn.init.kaiming_normal_(self.fc.weight)

    # xの形は(バッチサイズ, RNNへの入力数, チャンネル数, 解像度, 解像度)の5次元配列である必要がある
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        sequence_length = x.size()[1]
        # (batch_size, seq_len, frames, channel, colPixel, rowPixel)
        # -> (seq_len, batch_size, channel, frames, colPixel, rowPixel)
        x = x.permute(1, 0, 3, 2, 4, 5)
        x = torch.stack([torch.flatten(self.resnet18_3d(x[i]), 1) for i in range(sequence_length)])
        x = self.rnn(x)[0]
        x = self.fc(x)
        return x


if __name__ == '__main__':
    model = CNN3dRNN()
    input = torch.randn(2, 4, 16, 3, 80, 80)
    output = model(input)
    print(f'{output.shape = }')
