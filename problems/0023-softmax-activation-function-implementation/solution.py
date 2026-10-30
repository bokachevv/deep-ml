import torch

def softmax(scores: list[float]) -> list[float]:

    scores = torch.tensor(scores, dtype = torch.float32)
    max_score = torch.max(scores)
    exp_score = torch.exp(scores - max_score)
    soft_scores = (exp_score)/(torch.sum(exp_score))
    final_soft_scores = soft_scores.tolist()

    #softmax_torch = torch.softmax(scores, dim=-1)

    return [round(x,4) for x in final_soft_scores]