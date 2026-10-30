import torch
import torch.nn as nn

def train_neuron(features: torch.Tensor, labels: torch.Tensor, initial_weights: torch.Tensor, initial_bias: float, learning_rate: float, epochs: int) -> tuple[list[float], float, list[float]]:
    
    weights = torch.tensor(initial_weights, dtype=torch.float32, requires_grad=True)
    bias = torch.tensor(initial_bias, dtype=torch.float32, requires_grad=True)
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)

    Loss_values = []


    for _ in range(epochs):
        logits = features @ weights + bias
        sigmoid = torch.sigmoid(logits)
        Loss = torch.nn.functional.mse_loss(sigmoid, labels, reduction = 'mean')
        Loss_values.append(round(Loss.item(), 4))

        Loss.backward()

        with torch.no_grad():
            weights -= learning_rate * weights.grad
            bias -= learning_rate * bias.grad

            weights.grad.zero_()
            bias.grad.zero_()

    return [round(x, 4) for x in weights.tolist()], round(bias.item(), 4), Loss_values
