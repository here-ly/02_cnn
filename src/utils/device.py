import torch


def get_device() -> torch.device:
    try:
        if torch.mlu.is_available():
            return torch.device("mlu:0")
    except Exception:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
