import torch

class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, Q):
        Q_round = torch.round(input/Q)
        Q_q = Q_round * Q
        return Q_q
    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs, None