import torch
import torch.nn as nn
import torchvision.models as models
from skimage import color
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
from PIL import Image

#device cpu
device = torch.device('cpu')
# inception  ResNet
model_inception = models.inception_v3(pretrained=True).to(device)
model_inception.transform_input = False
model_inception.eval()

#off gradients for inception's model
for param in model_inception.parameters():
    param.requires_grad = False


class DeepKoalarization(nn.Module):
    def __init__(self, inception_res_net, batch_size: int):
        super(DeepKoalarization, self).__init__()
        # inception Res Net model
        self.__model = inception_res_net
        self.__batch_size = batch_size

        # encoders layers convs
        self.relu = nn.ReLU()
        self.e_conv1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64))
        self.e_conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128))
        self.e_conv3 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(128))
        self.e_conv4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256))
        self.e_conv5 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(256))
        self.e_conv6 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512))
        self.e_conv7 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512))
        self.e_conv8 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256))

        # fusion layer

        self.f_conv = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1), nn.BatchNorm2d(256))

        # decoder layers
        self.d_conv1 = nn.Sequential(nn.Conv2d(1256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128))
        self.d_upsampl1 = nn.Upsample(scale_factor=2)
        self.d_conv2 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64))
        self.d_conv3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64))
        self.d_upsampl2 = nn.Upsample(scale_factor=2)
        self.d_conv4 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32))
        self.d_conv5 = nn.Sequential(nn.Conv2d(32, 2, kernel_size=3, padding=1), nn.BatchNorm2d(2))
        self.d_upsampl3 = nn.Upsample(scale_factor=2)

    def __encoder(self, input_x: torch.tensor) -> torch.tensor:
        output = self.relu(self.e_conv1(input_x))
        output = self.relu(self.e_conv2(output))
        output = self.relu(self.e_conv3(output))
        output = self.relu(self.e_conv4(output))
        output = self.relu(self.e_conv5(output))
        output = self.relu(self.e_conv6(output))
        output = self.relu(self.e_conv7(output))
        output = self.relu(self.e_conv8(output))

        return output

    def __decoder(self, input_x: torch.tensor) -> torch.tensor:
        output = self.relu(self.d_conv1(input_x))
        output = self.d_upsampl1(output)
        output = self.relu(self.d_conv2(output))
        output = self.relu(self.d_conv3(output))
        output = self.d_upsampl2(output)
        output = self.relu(self.d_conv4(output))
        output = self.relu(self.d_conv5(output))
        output = self.d_upsampl3(output)

        return output

    def __get_features(self, features: torch.tensor, shape) -> torch.tensor:
        feature_mult = []

        for i in range(self.__batch_size):
            repeat = features[i].repeat(shape[2] * shape[3])
            x = repeat.view(shape[2], shape[3], 1000).transpose(1, 2).transpose(0, 1)
            feature_mult.append(x)

        return torch.stack(feature_mult)

    def forward(self, input_x: torch.tensor, input_x_features: torch.tensor):

        encoder = self.__encoder(input_x)
        fusion_conv = self.f_conv(encoder)
        features_inception = self.__model(input_x_features)
        fusion_output = torch.cat([fusion_conv, self.__get_features(features_inception, fusion_conv.shape)], dim=1)
        decoder = self.__decoder(fusion_output)
        return decoder



def normalize_data(data):
    data = data / 128
    return data
def get_lab(image, size):
    trans = transforms.CenterCrop(size)
    lab = color.rgb2lab(np.array(trans(image))/255)
    L, a, b = cv2.split(lab)
    return (L, a, b)

def transform_image(image, size: int, output_channels=3):
    trans = transforms.Compose([
    transforms.CenterCrop(size),
    transforms.Grayscale(num_output_channels=output_channels),
    transforms.ToTensor()
])
    return trans(image)


def create_images(image_model, l, size):
    a = image_model.detach().cpu().numpy()[0] * 128
    b = image_model.detach().cpu().numpy()[1] * 128
    lab = np.zeros([size[0], size[1], 3])
    lab[:, :, 0] = l
    lab[:, :, 1] = a
    lab[:, :, 2] = b

    return lab

if __name__ == '__main__':
    img = Image.open('img1.jpeg')
    size = (img.size[1], img.size[0])
    fusion_image = transform_image(img, size=size, output_channels=1)
    inception_image = transform_image(img, size=size)
    print(fusion_image.shape)
    print(inception_image.shape)
    L, _, _ = get_lab(img, size=size)

    weights = 'weights/deep_koala.pt'

    model = DeepKoalarization(model_inception, 1).to(device)

    model.load_state_dict(torch.load(weights, map_location=torch.device('cpu')))
    model.eval()

    output = model(fusion_image.unsqueeze(0), inception_image.unsqueeze(0))
    result_image = color.lab2rgb(create_images(output.squeeze(0), L, size=size))

    fig, axs = plt.subplots(1, 2, figsize=(10, 10))
    axs[0].imshow(img)
    axs[0].set_title('Original image')
    axs[1].imshow(result_image)
    axs[1].set_title('Predicted image')
    plt.show()