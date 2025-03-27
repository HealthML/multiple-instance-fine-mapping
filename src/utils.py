import torch

from src import models


def load_model(model_checkpoint):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    try:
        model_cls_str = torch.load(model_checkpoint, map_location='cpu')[
            'hyper_parameters'].get('model_class', 'DeepVEP')
        model_cls = getattr(models, model_cls_str)
        model = model_cls.load_from_checkpoint(model_checkpoint, map_location=device).eval()
    except TypeError as e:
        if 'object is not subscriptable' in str(e):
            model = torch.load(model_checkpoint).to(device).eval()
        else:
            raise e

    return model
