#pragma once
#include <torch/extension.h>
#include <tuple>

torch::Tensor 
ComputeFuncCPP(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& w,
    const torch::Tensor& x);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> 
ComputeFuncBackwardCPP(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& w,
    const torch::Tensor& x,
    const torch::Tensor& grad_out);