
import torch
import torchvision.utils as vutils
import torchvision.transforms as transforms
from PIL import Image
import os
from tqdm import tqdm



class Diffusion_model:
    def __init__(self, device, time_steps=1000, beta_start= 1e-4, beta_end=0.02):
        self.device = device
        self.time_steps = time_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = torch.linspace(beta_start, beta_end, time_steps).to(device)  # Linear beta schedule
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)


    # Generate noise under a normal distribution.
    def generate_noise(self,x):
        return torch.randn_like(x)

    #add noise to image
    def add_noise(self, x, t):
        # x is image shape [batch, 1, 32, 32]
        # t is time_step shape [batch]
        noise = self.generate_noise(x)

        alpha_hat_t = self.alpha_hat[t].view(-1, 1, 1, 1)  # reshape for broadcasting

        return torch.sqrt(alpha_hat_t) * x + torch.sqrt(1 - alpha_hat_t) * noise, noise

    # sampling. run through an image and generate noise based off of the time epoch. We start off with a full noisy image
    def sample(self, model, batch_size, in_channels, height, width):

        model.eval()

        with torch.no_grad():
            x = torch.randn(batch_size, in_channels, height, width).to(self.device)
            for step in tqdm((range(self.time_steps-1, 0, -1)), position=0):
                # current time. shape [batch]
                t = torch.full((batch_size,), step, dtype=torch.long).to(self.device)

                # running image through U-net to see what the noise is and we remove that
                noisy_image = x
                noise = model(noisy_image,t)

                # extra noise we add at the end to make sure the image still has some noise
                if step != 0:
                    add_noise = torch.randn_like(x).to(self.device)
                else:
                    add_noise = torch.zeroes_like(x).to(self.device)
                    print(f"yes {step}")

                # bunch of complex math (i dont understand) to remove noise
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]

                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * noise) + torch.sqrt(beta) * add_noise

        model.train()

        # this is from yter idk if this is needed.
        # x = (x.clamp(-1, 1) + 1) / 2
        # x = (x * 255).type(torch.uint8)

        return x

    # sampling but with classes instead
    def sample_classes(self, model, batch_size, in_channels, height, width, num_classes, guidance_scale = 3.5):

        model.eval()

        with torch.no_grad():
            x = torch.randn(batch_size, in_channels, height, width).to(self.device)
            for step in tqdm((range(self.time_steps - 1, 0, -1)), position=0):
                # current time. shape [batch]
                t = torch.full((batch_size,), step, dtype=torch.long).to(self.device)

                classes = torch.randint(0, num_classes, (batch_size,), dtype=torch.long).to(self.device)

                # running image through U-net to see what the noise is and we remove that
                noisy_image = x
                class_noise = model(noisy_image, t,classes)

                non_class_noise = model(noisy_image, t, None)

                noise = torch.lerp(non_class_noise,class_noise, guidance_scale)

                # extra noise we add at the end to make sure the image still has some noise
                if step != 0:
                    add_noise = torch.randn_like(x).to(self.device)
                else:
                    add_noise = torch.zeroes_like(x).to(self.device)
                    print(f"yes {step}")

                # bunch of complex math (i dont understand) to remove noise
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]

                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * noise) + torch.sqrt(
                    beta) * add_noise

        model.train()

        # this is from yter idk if this is needed.
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)

        return x


def save_images(images, path, nrow=4):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Convert from [-1, 1] to [0, 1] if needed
    if images.min() < 0:
        images = (images + 1) / 2.0

    grid = vutils.make_grid(images, nrow=nrow, padding=2)
    ndarr = grid.mul(255).clamp(0, 255).byte().cpu().numpy()
    ndarr = ndarr.transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)

    im = Image.fromarray(ndarr)
    im.save(path)


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())
