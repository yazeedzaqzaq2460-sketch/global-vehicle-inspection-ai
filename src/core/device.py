import torch


def get_device():
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return "cuda"

    print("Using CPU")
    return "cpu"


DEVICE = get_device()