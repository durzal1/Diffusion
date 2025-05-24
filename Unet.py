import torch
import torch.nn as nn
import torch.nn.functional as F


class Unet(nn.Module):

    def __init__(self, input_channels, output_channels, time_dim, device):
        super().__init__()
        self.device = device
        self.time_dim = time_dim


    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
                10000
                ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self,x,t):

        # Positionally encode everything
        pos_encoding = self.pos_encoding(t, self.time_dim)





batch = 16
height = 28
width = 28

tensor = torch.randn(batch, height,width)
print(tensor)

new_tesnor = Unet(tensor)
print(new_tesnor)