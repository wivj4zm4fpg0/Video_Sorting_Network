import torch
import torch.nn.functional as F
from torch import nn

from siamese_resnet_3d import generate_model


class SiameseCNN3D(nn.Module):
    def __init__(self, resnet_3d_pretrained_model='r3d18_KM_200ep.pth'):
        super().__init__()

        resnet18_3d = generate_model(18)
        checkpoint = torch.load(resnet_3d_pretrained_model)
        resnet18_3d.fc = nn.Linear(512, 1039)
        resnet18_3d.load_state_dict(checkpoint['state_dict'])
        resnet18_3d.siamese_cnn3d_init()

        self.resnet18_3d = resnet18_3d
        self.softmax = nn.Softmax(dim=1)

    def forward(self, search_videos: torch.Tensor, ref_videos: torch.Tensor) -> torch.Tensor:
        search_feat = self.resnet18_3d(search_videos.permute(0, 2, 1, 3, 4))
        ref_feat = self.resnet18_3d(ref_videos.permute(0, 2, 1, 3, 4))
        b, c, t, h, w = search_feat.size()
        match_out = F.conv3d(search_feat.view(-1, b * c, t, h, w), ref_feat, groups=b)
        match_out = match_out.view(b, -1)
        match_out = self.softmax(match_out)

        return match_out


if __name__ == '__main__':
    model = SiameseCNN3D()
    search_input = torch.randn(2, 64, 3, 70, 70)
    ref_input = torch.randn(2, 16, 3, 70, 70)
    output = model(search_input, ref_input)
    print(f'{output.shape=}')
    print(f'{output = }')
