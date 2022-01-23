import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F

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
