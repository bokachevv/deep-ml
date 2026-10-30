import torch
import torch.nn.functional as F

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> tuple[list[float], float]:
    features, labels, weights, bias = torch.tensor(features, dtype = torch.float32), torch.tensor(labels, dtype = torch.float32), torch.tensor(weights, dtype = torch.float32), torch.tensor(bias, dtype = torch.float32)

    logits = features @ weights + bias
    sigmoid = F.sigmoid(logits)

    MSE = torch.nn.functional.mse_loss(sigmoid, labels, reduction = 'mean')

    return sigmoid.tolist(), MSE.item()