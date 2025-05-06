import torch
from compute_func import compute
from params import Params

sample_size = 5000
num_epochs = 100

x = torch.randn(sample_size, 5, 1)
y = torch.randn(sample_size, 3, 1)

mse_loss = torch.nn.MSELoss()
params = Params()
optimizer = params.optimizer

for epoch in range(num_epochs):
    for i in range(sample_size):
        x_sample = x[i]
        y_sample = y[i]

        out_y = compute(x_sample, params)

        loss = mse_loss(out_y, y_sample)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")