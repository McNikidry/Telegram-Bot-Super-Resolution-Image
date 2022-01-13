from model import Generator, Discriminator
from data_preparation import DatasetSuperResolution
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

generator = Generator().to(device)
discriminator = Discriminator().to(device)

generator_loss = nn.MSELoss()
discriminator_loss = nn.BCEWithLogitsLoss()

generator_optimizer = torch.optim.AdamW(generator.parameters(),
                                        lr = 0.0003, betas = (0.5, 0.999))
discriminator_optimizer = torch.optim.AdamW(discriminator.parameters(),
                                            lr = 0.0003, betas = (0.5, 0.999))


def load_data(
        data_dir: str,
        batch_size: int = 20
):
    dataset = DatasetSuperResolution(data_dir)
    dataloader = DataLoader(dataset,
                            batch_size = batch_size,
                            shuffle = True,
                            num_workers = 2,
                            pin_memory = True)

    return dataloader


def train(
        train_loader,
        generator,
        discriminator,
        gen_loss,
        discr_loss,
        gen_optim,
        discr_optim,
        device,
        epochs: int
):
    generator.train()
    discriminator.train()

    for epoch in range(epochs):
        for lr, hr in train_loader:
            lr = lr.to(device)
            hr = hr.to(device)

            real_label = torch.full([lr.size(0), 1], 1.0, dtype = lr.dtype, device = device)
            fake_label = torch.full([lr.size(0), 1], 0.0, dtype = lr.dtype, device = device)

            sr = generator(lr)

            for p in discriminator.parameters():
                p.requires_grad = False

            discr_optim.zero_grad()

            hr_output = discriminator(hr)
            discriminator_hr_loss = discriminator_loss(hr_output, real_label)
            discriminator_hr_loss.backward()

            sr_output = discriminator(sr.detach())
            discriminator_sr_loss = discriminator_loss(sr_output, fake_label)
            discriminator_sr_loss.backward()

            discr_optim.step()

            for p in discriminator.parameters():
                p.requires_grad = False

            gen_optim.zero_grad()

            output = discriminator(sr)
            generator_loss_tr = generator_loss(output, hr)

            total_loss = generator_loss_tr + discriminator_sr_loss
            total_loss.backward()
            gen_optim.step()

    return discriminator_sr_loss.item(), generator_loss_tr.item(), total_loss.item(), sr, hr


class ContentMetric(nn.Module):

    def __init__(self):
        super(ContentMetric, self).__init__()
        vgg19 = models.vgg19(pretrained = True).eval()
        # Extract the thirty-sixth layer output in the VGG19 model as the content loss.
        self.feature_extractor = nn.Sequential(*list(vgg19.features.children())[:36])

        for parameters in self.feature_extractor.parameters():
            parameters.requires_grad = False

        self.register_buffer("mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, hr: torch.Tensor, sr: torch.Tensor):
        sr = sr.sub(self.mean).div(self.std)
        hr = hr.sub(self.mean).div(self.std)

        loss = F.l1_loss(self.feature_extractor(sr), self.feature_extractor(hr))

        return loss
