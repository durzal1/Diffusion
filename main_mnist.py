import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from Unet import Unet
from helper_functions import *

# Hyperparameters

width = 32
height = 32
time_dim = 256
epochs = 100
batch_size = 64
learning_rate = 1e-4
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
input_channels = 1
output_channels = 1
time_steps = 1000
save_every = 1

print(device)
# ---------- Data ----------
transform = transforms.Compose([
    transforms.Pad(2),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Setting up model
model = Unet(input_channels, output_channels, time_dim, device).to(device)
#
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
mse = nn.MSELoss()
diffusion = Diffusion_model(device)

# Training Loop
sampled = diffusion.sample(model, batch_size, input_channels, height, width)

for epoch in range(epochs):
    prog_bar = tqdm(dataloader)

    total_loss = 0
    num_steps = 0
    for step, (images, labels) in enumerate(dataloader):
        images = images.to(device)


        t_batch = torch.full((images.size(0),), torch.randint(0, time_steps, (1,)).item(), dtype=torch.long).to(device)

        noisy_image,generated_noise = diffusion.add_noise(images, t_batch)
        predicted_noise = model(noisy_image,t_batch)

        loss = mse(generated_noise, predicted_noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_steps += 1

        if (step + 1) % 100 == 0:
            print(f"Epoch: {epoch}, Step: {step}, Loss: {total_loss / num_steps:.4f}")

        if (step+1) % 500 == 0:
            sampled = diffusion.sample(model, batch_size, input_channels, height,width)
            save_images(sampled, f"results/sample_epoch_{epoch}_step_{step}.png")

    print(f"Epoch: {epoch} Loss: {total_loss/num_steps:.4f}")

    # Sample images
