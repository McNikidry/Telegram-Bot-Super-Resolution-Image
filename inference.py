import torch
from model import Generator

def inference(path_to_model, lr_image):
    model = Generator().to('cpu')
    model.load_state_dict(torch.load(path_to_model, map_location = torch.device('cpu')))
    model.eval()
    output = model(lr_image)
    return output
