loader_conf = {
    'train_images_dir': '/data/train_set/',
    'valid_images_dir': '/data/valid_set/',
    'batch_size': 10,

}

optimizers_conf = {
    'generator_optimizer': [3e-4, (0.5, 0.999)],
    'discriminator_optimizer': [3e-4, (0.5, 0.999)]
}

lr_schedule_conf = {
    'step_size': 10,
    'gamma': 0.1
}

loss_conf = {
    'mse_loss': 1,
    'content_loss': 1,
    'adversarial_loss': 1e-3
}

path_to_models_srgan = {
    'generator': 'weights/srgan/generator/',
    'discriminator': 'weights/srgan/discriminator/'
}

epochs = 40