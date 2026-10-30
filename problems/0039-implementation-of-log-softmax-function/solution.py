import torch
import numpy as np
from typing import List, Union

def log_softmax(scores: Union[List[float], np.ndarray]) -> np.ndarray:

    if not isinstance(scores, torch.Tensor):
        scores_tensor = torch.tensor(scores, dtype=torch.float32)
    else:
        scores_tensor = scores.float()
        
    max_score = torch.max(scores_tensor)
    exp_score = torch.exp(scores_tensor - max_score)
    log_probs = torch.log(exp_score / torch.sum(exp_score))
 
    return np.round(log_probs.detach().numpy(), 4)

