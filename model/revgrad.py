import torch
from torch import nn

# source: https://github.com/tadeephuy/GradientReversal
class GradReversalF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = - alpha*grad_output
        return grad_input, None


class GradReversal(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, alpha):
        alpha = torch.tensor(alpha, requires_grad=False)
        return GradReversalF.apply(x, alpha)