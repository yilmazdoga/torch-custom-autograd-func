import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
cpp_parent_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cpp-compute-func")
_CPP = load(
    name='cpp_compute_func',
    sources=[
        os.path.join(cpp_parent_dir, "compute_func.cpp"),
        os.path.join(cpp_parent_dir, "ext.cpp")],
    )


def compute_func(a, b, w, x):
    return _ComputeFunc.apply(a, b, w, x)


class _ComputeFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, w, x):  
        # Pyhton implementation:
        # y = xˆT * wˆT * w * x + a * x + b
        # out = (x.transpose(0, 1) @ w.transpose(0, 1) @ w @ x) + (a @ x) + b

        # C++ implementation:
        out = _CPP.compute_func(a, b, w, x)

        # CUDA implementation:
        # out = _CUDA.compute_func(x, w, a, b)

        ctx.x = x
        ctx.save_for_backward(a, b, w)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x = ctx.x
        a, b, w = ctx.saved_tensors

        # Pyhton implementation:
        # # dL/da = dL/dy * dy/da = grad_out * x
        # grad_a = grad_out @ x.transpose(0, 1)
        
        # # dL/db = dL/dy * dy/db = grad_out * 1
        # grad_b = grad_out 
        
        # # dL/dw = dL/dy * dy/dw = grad_out * 2wxxˆT
        # grad_w = 2 * (grad_out * (w @ x)) @ x.transpose(0, 1)

        # C++ implementation:
        (grad_a, grad_b, grad_w) = _CPP.compute_func_backward(a, b, w, x, grad_out)

        # CUDA implementation:
        # grads = _CUDA.compute_func_backward(a, b, w, x, grad_out)

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