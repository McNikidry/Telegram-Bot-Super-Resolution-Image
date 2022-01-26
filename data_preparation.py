import torch
import numpy as np
import os
from torchvision import transforms
from torch.utils.data import Dataset
from skimage import io
from torchvision.transforms.functional import InterpolationMode as IMode


class DatasetSuperResolution(Dataset):

    def __init__(
            self,
            path_to_data: str,
            mode: str = 'train',
            image_size: int = 1080,
            upscale_factor: int = 4

    ):
        super(DatasetSuperResolution, self).__init__()

        self.files = [os.path.join(path_to_data, x) for x in os.listdir(path_to_data)]

        if mode == "train":
            self.hr_transforms = transforms.Compose([
                transforms.RandomCrop(image_size, pad_if_needed = True)
            ])
        else:
            self.hr_transforms = transforms.CenterCrop(image_size)

        self.lr_transforms = transforms.Resize(
            image_size // upscale_factor,
            interpolation = IMode.BICUBIC,
            antialias = True
        )

    def __getitem__(self, _index: int) -> [torch.Tensor, torch.Tensor]:
        image = io.imread(self.files[_index])
        image = transforms.ToTensor()(image)

        hr_image = self.hr_transforms(image)
        lr_image = self.lr_transforms(hr_image)

        return lr_image, hr_image

    def __len__(self) -> int:
        return len(self.files)


class DatasetNoiseReduction(Dataset):

    def __init__(
            self,
            path_to_data: str,
            mode: str = 'train',
            image_size: int = 1080,
            mean: float = 0.,
            std: float = 0.1

    ):
        super(DatasetNoiseReduction, self).__init__()

        self.files = [os.path.join(path_to_data, x) for x in os.listdir(path_to_data)]

        if mode == "train":
            self.hr_transforms = transforms.Compose([
                transforms.RandomCrop(image_size, pad_if_needed = True)
            ])
        else:
            self.hr_transforms = transforms.CenterCrop(image_size)

        self.lr_transforms = transforms.Compose([
            AddGaussianNoise(mean, std ** 0.5)]
        )

    def __getitem__(self, _index: int) -> [torch.Tensor, torch.Tensor]:
        image = io.imread(self.files[_index])
        image = transforms.ToTensor()(image)

        hr_image = self.hr_transforms(image)
        lr_image = self.lr_transforms(hr_image)

        return lr_image, hr_image

    def __len__(self) -> int:
        return len(self.files)


class AddGaussianNoise():
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)