import torch

def adam_optimizer(f, grad, x0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=10) -> torch.Tensor:
    x = torch.tensor(x0, dtype = torch.float32)

    momentum, rms = torch.zeros_like(x), torch.zeros_like(x)


    for t in range(1, num_iterations+1):
        grads = grad(x.tolist())
        grads = torch.tensor(grads, dtype = torch.float32)

        momentum = beta1 * momentum + (1 - beta1) * grads
        rms = beta2 * rms + (1 - beta2) * grads**2

        momentum_corr = momentum / (1- beta1**t)
        rms_corr = rms / (1 - beta2**t)

        x -= learning_rate * (momentum_corr)/ (torch.sqrt(rms_corr) + epsilon)
    
    return x