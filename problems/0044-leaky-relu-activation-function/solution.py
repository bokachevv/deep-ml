import torch

def leaky_relu(z: torch.Tensor, alpha: float = 0.01) -> torch.Tensor:
    
    z = torch.tensor(z, dtype = torch.float32)

    if alpha is not None:
        return torch.maximum(z*alpha, z)
    else:
        return torch.maximum(torch.tensor(0, dtype = torch.float32), z)