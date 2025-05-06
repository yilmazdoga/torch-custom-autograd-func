import torch
import torch.nn as nn


def compute_func(a, b, w, x):
    return _ComputeFunc.apply(a, b, w, x)


class _ComputeFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, w, x):  
        # y = xˆT * wˆT * w * x + a * x + b
        out = (x.transpose(0, 1) @ w.transpose(0, 1) @ w @ x) + (a @ x) + b

        ctx.x = x
        ctx.save_for_backward(a, b, w)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x = ctx.x
        a, b, w = ctx.saved_tensors

        # dL/da = dL/dy * dy/da = grad_out * x
        grad_a = grad_out @ x.transpose(0, 1)
        
        # dL/db = dL/dy * dy/db = grad_out * 1
        grad_b = grad_out 
        
        # dL/dw = dL/dy * dy/dw = grad_out * 2wxxˆT
        grad_w = 2 * (grad_out * (w @ x)) @ x.transpose(0, 1)

        grads = (
            grad_a,
            grad_b,
            grad_w,
            None,
        )

        return grads 


class ComputeFunc(nn.Module):
    def __init__(self, x):
        self.x = x
        super(ComputeFunc, self).__init__()

    def forward(self, a, b, w):
        return compute_func(a, b, w, self.x)