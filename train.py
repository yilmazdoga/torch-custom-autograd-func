import torch
import random
import numpy as np
from compute_func import compute
from params import Params

seed = 18
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

sample_size = 5000
num_epochs = 100
batch_size = 10

x = torch.randn(sample_size, 5, 1)
y = torch.randn(sample_size, 3, 1)

mse_loss = torch.nn.MSELoss()
params = Params()
optimizer = params.optimizer

mode = 'both'  # 'gradient_accum' or 'loss_accum' or 'both'

for epoch in range(num_epochs):
    for i in range(sample_size // batch_size):
        x_sample = x[i * batch_size: (i + 1) * batch_size]
        y_sample = y[i * batch_size: (i + 1) * batch_size]
        batch_grads_a = []
        batch_grads_b = []
        batch_grads_w = []
        if mode == 'gradient_accum' or mode == 'both':
            batch_loss = 0.0
            for x_item, y_item in zip(x_sample, y_sample):
                out_y = compute(x_item, params)
                loss = mse_loss(out_y, y_item)
                batch_loss += loss
                loss.backward()
                
                batch_grads_a.append(params.get_a.grad)
                batch_grads_b.append(params.get_b.grad)
                batch_grads_w.append(params.get_w.grad)
            
            batch_grads_a = torch.stack(batch_grads_a)
            batch_grads_a = torch.mean(batch_grads_a, dim=0)
            batch_grads_b = torch.stack(batch_grads_b)
            batch_grads_b = torch.mean(batch_grads_b, dim=0)
            batch_grads_w = torch.stack(batch_grads_w)
            batch_grads_w = torch.mean(batch_grads_w, dim=0)

            params.get_a.grad = batch_grads_a
            params.get_b.grad = batch_grads_b
            params.get_w.grad = batch_grads_w
            batch_loss = batch_loss / batch_size

        if mode == 'loss_accum' or mode == 'both':
            if mode == 'both':
                optimizer.zero_grad()
                
            batch_loss = 0.0
            for x_item, y_item in zip(x_sample, y_sample):
                out_y = compute(x_item, params)
                loss = mse_loss(out_y, y_item)
                batch_loss += loss
            batch_loss = batch_loss / batch_size
            batch_loss.backward()

        if mode == 'both':
            if not torch.allclose(params.get_a.grad, batch_grads_a):
                print("Gradient mismatch for a")
                exit(1)
            if not torch.allclose(params.get_b.grad, batch_grads_b):
                print("Gradient mismatch for b")
                exit(1)
            if not torch.allclose(params.get_w.grad, batch_grads_w):
                print("Gradient mismatch for w")
                exit(1)

        optimizer.step()
        optimizer.zero_grad()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {batch_loss.item():.4f}")