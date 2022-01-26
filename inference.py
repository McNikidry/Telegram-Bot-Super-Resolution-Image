import torch
from model import Generator
import time

def inference(path_to_model, lr_image):
    model = Generator().to('cpu')
    model.load_state_dict(torch.load(path_to_model, map_location = torch.device('cpu')))
    model.eval()
    model_int8 = torch.quantization.quantize_dynamic(
        model,  # the original model
        {torch.nn.Linear,
         torch.nn.Conv2d,
         torch.nn.ReLU},  # a set of layers to dynamically quantize
        dtype=torch.qint8)
    output = model_int8(lr_image)
    return output
