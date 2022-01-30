import torch


def inference(path_to_model_sr: None, path_to_model_denoise: None, sr_model: None, denoise_model: None, image):

    if sr_model is None:
        model = denoise_model
        path_to_model = path_to_model_denoise
    elif denoise_model is None:
        model = sr_model
        path_to_model = path_to_model_sr
    else:
        model_sr = sr_model
        model_den = denoise_model

    if sr_model is None or denoise_model is None:
        model = model.to('cpu')
        model.load_state_dict(torch.load(path_to_model, map_location = torch.device('cpu')))
        model.eval()
        model_int8 = torch.quantization.quantize_dynamic(
            model,  # the original model
            {torch.nn.Linear,
             torch.nn.Conv2d,
             torch.nn.ReLU},  # a set of layers to dynamically quantize
            dtype = torch.qint8)
        output = model_int8(image)
    else:
        model = model_sr.to('cpu')
        model.load_state_dict(torch.load(path_to_model_sr, map_location = torch.device('cpu')))
        model.eval()
        model_int8 = torch.quantization.quantize_dynamic(
            model,  # the original model
            {torch.nn.Linear,
             torch.nn.Conv2d,
             torch.nn.ReLU},  # a set of layers to dynamically quantize
            dtype = torch.qint8)
        output_sr = model_int8(image)

        model = model_den.to('cpu')
        model.load_state_dict(torch.load(path_to_model_denoise, map_location = torch.device('cpu')))
        model.eval()
        model_int8 = torch.quantization.quantize_dynamic(
            model,  # the original model
            {torch.nn.Linear,
             torch.nn.Conv2d,
             torch.nn.ReLU},  # a set of layers to dynamically quantize
            dtype = torch.qint8)
        output = model_int8(output_sr)

    return output
