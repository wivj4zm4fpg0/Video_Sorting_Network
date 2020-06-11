import torch
from torch import nn
# from torchvision.models import resnet18
# from torchvision.models.resnet import BasicBlock
from torchvision.models.alexnet import alexnet


class CNN_LSTM(nn.Module):
    def __init__(self, class_num: int = 101, bidirectional: bool = True, pretrained: bool = True,
                 task: str = 'classification'):
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

        # resnet18_modules = [module for module in (resnet18(pretrained=pretrained).modules())][1:-1]
        # resnet18_modules_cut = resnet18_modules[0:4]
        # resnet18_modules_cut.extend(
        #     [module for module in resnet18_modules if type(module) == nn.Sequential and type(module[0]) == BasicBlock])
        # resnet18_modules_cut.append(resnet18_modules[-1])
        # self.cnn = nn.Sequential(*resnet18_modules_cut)
        # cnn_last_dim = 512

        alex_model = alexnet(pretrained=pretrained)
        self.cnn = nn.Sequential(*alex_model.features, alex_model.avgpool)
        cnn_last_dim = 9216

        # self.pre_fc1 = nn.Linear(cnn_last_dim, 4096)
        # self.pre_relu1 = nn.ReLU(inplace=True)
        # self.pre_fc2 = nn.Linear(4096, 4096)
        # self.pre_relu2 = nn.ReLU(inplace=True)
        # nn.init.kaiming_normal_(self.pre_fc1.weight)
        # nn.init.kaiming_normal_(self.pre_fc2.weight)
        # cnn_last_dim = 4096

        lstm_dim = 512
        num_layers = 2
        if bidirectional:
            self.gru = nn.GRU(cnn_last_dim, int(lstm_dim / 2), bidirectional=True, num_layers=num_layers,
                              batch_first=batch_first)
        else:
            self.gru = nn.GRU(cnn_last_dim, lstm_dim, bidirectional=True, num_layers=num_layers,
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
        sequence_length = x.shape[1]
        x = x.permute(1, 0, 2, 3, 4)  # (batch_size, seq_len, img) -> (seq_len, batch_size, img)
        x = torch.stack([torch.flatten(self.resnet18(x[i]), 1) for i in range(sequence_length)])
        # x = torch.stack(
        #     [self.pre_relu2(self.pre_fc2(self.pre_relu1(self.pre_fc1(torch.flatten(self.cnn(x[i]), 1))))) for i in
        #      range(sequence_length)])
        # x = torch.stack(
        #     [self.pre_fc2(self.pre_fc1(torch.flatten(self.resnet18(x[i]), 1))) for i in range(sequence_length)])
        # output_shape -> (seq_len, batch_size, data_size), lstm.batch_first -> Flase
        # x = self.lstm(x)[0]
        x = self.gru(x)[0]
        x = self.fc(x)
        return x

    def sorting_forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        # シーケンスでバッチ処理をする
        x = torch.stack([torch.flatten(self.cnn(x[i]), 1) for i in range(batch_size)])
        # x = torch.stack(
        #     [self.pre_fc2(self.pre_fc1(torch.flatten(self.resnet18(x[i]), 1))) for i in range(batch_size)])
        # output_shape -> (batch_size, seq_len, data_size), lstm.batch_first -> True
        # x = self.lstm(x)[0]
        x = self.gru(x)[0]
        x = self.fc(x)
        return x


if __name__ == '__main__':
    model = CNN_LSTM()
    input = torch.randn(2, 4, 3, 256, 256)
    output = model(input)
    print(f'{output.shape=}')
