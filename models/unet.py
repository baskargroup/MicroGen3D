import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce


class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = True) -> None:
        super().__init__()
        '''
        standard ResNet style convolutional block
        '''
        self.same_channels = in_channels==out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, 1, 1),
            nn.InstanceNorm3d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 3, 1, 1),
            nn.InstanceNorm3d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # this adds on correct residual in case channels have increased
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2 
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2
    
class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        self.model = nn.Sequential(
            ResidualConvBlock(in_channels, out_channels, is_res=True), 
            nn.MaxPool3d(2)
        )

    def forward(self, x):
        return self.model(x)

class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2, padding=0, output_padding=0),
            ResidualConvBlock(out_channels, out_channels, is_res=True),
            ResidualConvBlock(out_channels, out_channels, is_res=True),
        )

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        return self.model(x)

class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)


class ContextUnet_3D_2lvls(nn.Module):
    def __init__(self, in_channels, out_channels, n_feat=256, context_dim=2):
        super(ContextUnet_3D_2lvls, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_feat = n_feat

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool3d(1), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2 * n_feat)
        self.timeembed2 = EmbedFC(1, 1 * n_feat)

        self.contextembed1 = EmbedFC(context_dim, 2 * n_feat)
        self.contextembed2 = EmbedFC(context_dim, 1 * n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose3d(2 * n_feat, 2 * n_feat, 1, 1),
            nn.GroupNorm(8, 2 * n_feat),
            nn.LeakyReLU(0.2)
        )

        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv3d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.Tanh(),
            nn.Conv3d(n_feat, self.out_channels, 3, 1, 1)
        )

    def forward(self, x, t, context):
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)

        hiddenvec = self.to_vec(down2)  # converts channels to vector with average pooling

        # embed context, time step
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1, 1)

        cemb1 = self.contextembed1(context).view(-1, self.n_feat * 2, 1, 1, 1)
        cemb2 = self.contextembed2(context).view(-1, self.n_feat, 1, 1, 1)

        up1 = self.up0(hiddenvec)
        up2 = self.up1(up1 + temb1 + cemb1, down2)  # add and multiply embeddings
        up3 = self.up2(up2 + temb2 + cemb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out