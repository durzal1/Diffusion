import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, residual=False):
        super().__init__()
        self.residual = residual
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, out_channels),
            nn.GELU()
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.conv(x))
        return self.conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, embedding_dimension):
        super().__init__()

        self.conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, False),
            DoubleConv(out_channels, out_channels, True)

            # This is what YTber does
            # DoubleConv(in_channels, in_channels, True),
            # DoubleConv(in_channels, out_channels, False)
        )

        self.time_embedding = nn.Sequential(
            nn.SELU(),
            nn.Linear(embedding_dimension, out_channels)
        )

    def forward(self,x,t):
        x = self.conv(x)
        t = self.time_embedding(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])


        return x + t

class Up(nn.Module):

    def __init__(self, in_channels, out_channels, embedding_dimension):
        self. up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, out_channels, False),
            DoubleConv(out_channels, out_channels, True)

            # This is what YTber does
            # DoubleConv(in_channels, in_channels, True),
            # DoubleConv(in_channels, out_channels, False)
        )
        self.time_embedding = nn.Sequential(
            nn.SELU(),
            nn.Linear(embedding_dimension, out_channels)
        )

    def forward(self,x,past_x, t):

        x = self.up(x)
        x = torch.cat([past_x, x], dim=1)
        x = self.conv(x)
        t = self.time_embedding(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])

        return x + t

# I didn't implement this cuz it looks really confusing and isn't the focus of this project anyway
class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)

class Unet(nn.Module):

    def __init__(self, input_channels, output_channels, time_dim, device):
        super().__init__()
        self.device = device
        self.time_dim = time_dim

        self.double1 = DoubleConv(input_channels, 64)
        self.down1 = Down(64, 128, time_dim)
        self.sa1 = SelfAttention(128, 32)
        self.down2 = Down(128, 256, time_dim)
        self.sa2 = SelfAttention(256, 64)
        self.down3 = Down(256, 256, time_dim)
        self.sa3 = SelfAttention(256,8)

        self.bottle = nn.Sequential(
            DoubleConv(256, 512),
            DoubleConv(512, 512),
            DoubleConv(512, 256),
        )

        self.up1 = Up(256, 128, time_dim)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(256, 64, time_dim)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(128, 64, time_dim)
        self.sa6 = SelfAttention(64, 64)
        self.out = nn.Conv2d(64, output_channels, kernel_size=1)

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

        # Downscale
        x1 = self.double1(x)
        x2 = self.down1(x1,pos_encoding)
        x2 = self.sa1(x2)
        x3 = self.down2(x2,pos_encoding)
        x3 = self.sa2(x3)
        x4 = self.down3(x3,pos_encoding)
        x4 = self.sa3(x4)

        # Bottle neck (Just some double conv layers)
        x4 = self.bottle(x4)


        # Upscale
        x = self.up1(x4,x3,pos_encoding)
        x = self.sa4(x)
        x = self.up2(x,x2,pos_encoding)
        x = self.sa5(x)
        x = self.up3(x,x1,pos_encoding)
        x = self.sa6(x)

        x = self.out(x)

        return x





batch = 16
height = 28
width = 28
embedding_dimension = 64

tensor = torch.randn(batch, height,width)
print(tensor)

new_tesnor = Unet(tensor)
print(new_tesnor)