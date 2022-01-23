from torch.optim import lr_scheduler
from Content_loss import ContentMetric
from model import Generator, Discriminator
from data_preparation import DatasetSuperResolution
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from configs import *


def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    ## Models, Losses and Optimizers definition
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    generator_loss = nn.MSELoss()
    discriminator_loss = nn.BCEWithLogitsLoss()

    generator_optimizer = torch.optim.AdamW(generator.parameters(),
                                            lr = optimizers_conf['generator_optimizer'][0],
                                            betas = optimizers_conf['generator_optimizer'][1])
    discriminator_optimizer = torch.optim.AdamW(discriminator.parameters(),
                                                lr = optimizers_conf['discriminator_optimizer'][0],
                                                betas = optimizers_conf['discriminator_optimizer'][1])

    ## Upload data
    train_loader = load_data(loader_conf['train_images_dir'],
                             loader_conf['batch_size'],
                             'train')
    valid_loader = load_data(loader_conf['valid_images_dir'],
                             loader_conf['batch_size'],
                             'valid')

    ## Train loop
    train_mse_loss, train_content_loss, train_discriminator_sr_loss, \
    train_discriminator_hr_loss, validation_mse_losses = train(
        train_loader,
        valid_loader,
        generator,
        discriminator,
        generator_loss,
        discriminator_loss,
        generator_optimizer,
        discriminator_optimizer,
        device,
        epochs
    )
    return train_mse_loss, train_content_loss, train_discriminator_sr_loss, \
           train_discriminator_hr_loss, validation_mse_losses


def load_data(
        data_dir: str,
        batch_size: int = 20,
        mode: str = 'train'
):
    dataset = DatasetSuperResolution(path_to_data = data_dir, mode = mode)
    dataloader = DataLoader(dataset,
                            batch_size = batch_size,
                            shuffle = True)

    return dataloader


def train(
        train_loader,
        valid_loader,
        generator,
        discriminator,
        generator_loss,
        discriminator_loss,
        gen_optim,
        discr_optim,
        device,
        epochs: int
):
    ## Losses declaration
    train_mse_loss = []
    train_content_loss = []
    train_discriminator_sr_loss = []
    train_discriminator_hr_loss = []
    validation_mse_losses = []

    content_loss = ContentMetric().to(device)
    discriminator.train()
    scheduler_dicr = lr_scheduler.StepLR(discr_optim,
                                         step_size = lr_schedule_conf['step_size'],
                                         gamma = lr_schedule_conf['gamma'])
    scheduler_gener = lr_scheduler.StepLR(gen_optim,
                                          step_size = lr_schedule_conf['step_size'],
                                          gamma = lr_schedule_conf['gamma'])

    for epoch in range(epochs):
        generator.train()
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
            train_discriminator_hr_loss.append(discriminator_hr_loss.item())

            sr_output = discriminator(sr.detach())
            discriminator_sr_loss = discriminator_loss(sr_output, fake_label)
            train_discriminator_sr_loss.append(discriminator_sr_loss.item())

            total_discriminator_loss = discriminator_hr_loss + train_discriminator_hr_loss
            total_discriminator_loss.backward()

            discr_optim.step()

            for p in discriminator.parameters():
                p.requires_grad = False

            gen_optim.zero_grad()

            output = discriminator(sr)
            mse_loss = generator_loss(output, hr.detach())
            train_mse_loss.append(mse_loss)
            c_loss = content_loss(sr, hr.detach())
            train_content_loss.append(c_loss.item())
            adversarial_loss = loss_conf['adversarial_loss'] * discriminator_loss(output, real_label)

            total_loss_generator = loss_conf['mse_loss'] * mse_loss + \
                                   loss_conf['content_loss'] * c_loss + \
                                   loss_conf['adversarial_loss'] * adversarial_loss
            total_loss_generator.backward()
            gen_optim.step()

        scheduler_dicr.step()
        scheduler_gener.step()

        generator.eval()
        for x, y in valid_loader:
            with torch.no_grad():
                prediction = generator(x.to(device))
                loss = generator_loss(prediction, y.to(device))
                if epoch != 0:
                    if loss.item() < validation_mse_losses[-1]:
                        torch.save(discriminator.state_dict(), path_to_models_srgan['discriminator'])
                        torch.save(generator.state_dict(), path_to_models_srgan['generator'])
                validation_mse_losses.append(loss.item())

    return train_mse_loss, train_content_loss, train_discriminator_sr_loss, \
           train_discriminator_hr_loss, validation_mse_losses


if __name__ == "__main__":
    main()
