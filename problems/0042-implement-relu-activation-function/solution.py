import torch
import torch.nn.functional as F

def relu(z: float) -> torch.Tensor:
    z = torch.tensor(z, dtype=torch.float32)
    return F.relu(z)