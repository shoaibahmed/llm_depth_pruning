import torch


def get_num_model_params(model):
    return sum(
        [p.numel() * 2 if p.is_complex() else p.numel() for p in model.parameters() if p.requires_grad]
    )
