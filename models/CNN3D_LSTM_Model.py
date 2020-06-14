import torch
from torch import nn

from models.resnet_3d import generate_model


class CNN3D_LSTM(nn.Module):
    def __init__(self, class_num: int = 101, bidirectional: bool = True, pretrained: bool = True,
                 task: str = 'classification', resnet_3d_pretrained_model='r3d18_KM_200ep.pth'):
        super().__init__()

        assert task == 'classification' or task == 'sorting'
        batch_first = None
        if task == 'classification':
            self.forward_method = self.classification_forward
            # self.forward = self.classification_forward
            batch_first = False
        elif task == 'sorting':
            self.forward_method = self.sorting_forward
            # self.forward = self.sorting_forward
            batch_first = True

        resnet18_3d = generate_model(18)
        checkpoint = torch.load(resnet_3d_pretrained_model)
        resnet18_3d.fc = nn.Linear(512, 1039)
        resnet18_3d.load_state_dict(checkpoint['state_dict'])

        # resnet18_3d_modules = [module for module in resnet18_3d.modules()][1:-1]
        # resnet18_3d_modules_cut = resnet18_3d_modules[0:4]
        # resnet18_3d_modules_cut.extend([module for module in resnet18_3d_modules if
        #                                 type(module) == nn.Sequential and type(module[0]) == BasicBlock])
        # resnet18_3d_modules_cut.append(resnet18_3d_modules[-1])
        # self.resnet18_3d = nn.Sequential(*resnet18_3d_modules_cut)

        self.resnet18_3d = resnet18_3d

        resnet18_3d_last_dim = 512

        lstm_dim = 512
        if bidirectional:
            self.lstm = nn.LSTM(resnet18_3d_last_dim, int(lstm_dim / 2), bidirectional=True, num_layers=2,
                                batch_first=batch_first)
        else:
            self.lstm = nn.LSTM(resnet18_3d_last_dim, lstm_dim, bidirectional=False, num_layers=2,
                                batch_first=batch_first)

        self.fc = nn.Linear(lstm_dim, class_num)
        nn.init.kaiming_normal_(self.fc.weight)

    # xの形は(バッチサイズ, RNNへの入力数, チャンネル数, 解像度, 解像度)の5次元配列である必要がある
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # (バッチサイズ x RNNへの入力数, チャンネル数, 解像度, 解像度)の4次元配列に変換する
        # x = x.view(batch_size * sequence_length, x.shape[2], x.shape[3], x.shape[4])
        # x = self.resnet18(x)
        # x = x.view(batch_size, sequence_length, -1)
        # output_shape -> (batch_size, seq_len, data_size)

        # resnet18_last_dim = 512
        # fs = torch.zeros(batch_size, sequence_length, resnet18_last_dim).cuda()
        # for i in range(batch_size):
        #     cnn = self.resnet18(x[i])
        #     cnn = torch.flatten(cnn, 1)
        #     fs[i, :, :] = cnn
        # x = fs

        return self.forward_method(x)

    def classification_forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence_length = x.size()[1]
        # (batch_size, seq_len, frames, channel, colPixel, rowPixel)
        # -> (seq_len, batch_size, channel, frames, colPixel, rowPixel)
        x = x.permute(1, 0, 3, 2, 4, 5)
        x = torch.stack([torch.flatten(self.resnet18_3d(x[i]), 1) for i in range(sequence_length)])
        # output_shape -> (seq_len, batch_size, data_size), lstm.batch_first -> Flase
        x = self.lstm(x)[0]
        x = self.fc(x)
        return x

    def sorting_forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size()[0]
        # シーケンスでバッチ処理をする
        x = torch.stack([torch.flatten(self.resnet18_3d(x[i]), 1) for i in range(batch_size)])
        # output_shape -> (batch_size, seq_len, data_size), lstm.batch_first -> True
        x = self.lstm(x)[0]
        x = self.fc(x)
        return x


if __name__ == '__main__':
    model = CNN3D_LSTM()
    input = torch.randn(2, 4, 16, 3, 80, 80)
    output = model(input)
    print(f'{output.shape=}')