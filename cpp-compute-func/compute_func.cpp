#include <math.h>
#include <vector>
#include <torch/extension.h>


torch::Tensor 
ComputeFuncCPP(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& w,
    const torch::Tensor& x) 
    {
    // y = xˆT * wˆT * w * x + a * x + b
    torch::Tensor out = torch::matmul(x.transpose(0, 1), w.transpose(0, 1));
    out = torch::matmul(out, w);
    out = torch::matmul(out, x);
    out = out + torch::matmul(a, x);
    out = out + b;

    return out;
    }

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> 
ComputeFuncBackwardCPP(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& w,
    const torch::Tensor& x,
    const torch::Tensor& grad_out) 
    {
        // dL/da = dL/dy * dy/da = grad_out * x
        torch::Tensor grad_a = torch::matmul(grad_out, x.transpose(0, 1));
        
        // dL/db = dL/dy * dy/db = grad_out * 1
        torch::Tensor grad_b = grad_out;
        
        // dL/dw = dL/dy * dy/dw = grad_out * 2wxxˆT
        torch::Tensor grad_w = 2 * torch::matmul(grad_out * torch::matmul(w, x), x.transpose(0, 1));
        
        return std::make_tuple(grad_a, grad_b, grad_w);
    }