from random import uniform

from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F


class VideoRandomCrop(transforms.RandomCrop):
    def __call__(self, img: Image) -> Image:
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)

        i, j, h, w = self.ijhw

        return F.crop(img, i, j, h, w)

    def set_param(self, img: Image):
        self.ijhw: Tuple = transforms.RandomCrop.get_params(img, self.size)


class VideoRandomRotation(transforms.RandomRotation):
    def __call__(self, img: Image) -> Image:
        angle = self.degree

        return F.rotate(img, angle, self.resample, self.expand, self.center, self.fill)

    def set_degree(self, degree: int = 360):
        self.degree = uniform(0, degree)
