import torch

def sigmoid(z: float) -> float:

    z = torch.tensor(z, dtype = torch.float32)
    sigmoid = (1.0/(1.0+torch.exp(-z))).item()
    # sigmoid_torch = (torch.sigmoid(z)).item()

    return round(sigmoid,4)