import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import rearrange, repeat

class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(in_channels, affine=True),
            nn.ReLU(),
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(in_channels, affine=True),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = self.block(x)
        x = self.relu(x + residual)
        return x


class Encoder(nn.Module):
    def __init__(self, latent_dim=3, max_channels=512, hidden_dim=256):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.conv_layers = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=3, stride=1, padding=1), 
            nn.InstanceNorm3d(64, affine=True),
            ResBlock(64),
            
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1), 
            nn.InstanceNorm3d(128, affine=True),
            ResBlock(128),
            
            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1), 
            nn.InstanceNorm3d(256, affine=True),
            ResBlock(256),

            nn.Conv3d(256, max_channels, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(max_channels, affine=True),
            ResBlock(max_channels),
            
            nn.Conv3d(max_channels, latent_dim, kernel_size=1, stride=1, padding=0), 
            nn.InstanceNorm3d(latent_dim, affine=True),
            ResBlock(latent_dim),
        )

        self.conv_mu = nn.Conv3d(latent_dim, latent_dim, kernel_size=1)
        self.conv_logvar = nn.Conv3d(latent_dim, latent_dim, kernel_size=1)

    def forward(self, x):
        x = self.conv_layers(x)
        mu = self.conv_mu(x)
        logvar = self.conv_logvar(x)

        return mu, logvar
class Decoder(nn.Module):
    def __init__(self, latent_dim=3, max_channels=512):
        super(Decoder, self).__init__()


        channels = [latent_dim, max_channels, 256, 128, 64]
        
        self.convs = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        self.instnorms = nn.ModuleList()

        for i in range(len(channels)-1):
            self.convs.append(nn.ConvTranspose3d(channels[i] , channels[i+1], kernel_size=3 if i else 1, stride=2 if i else 1, padding=1 if i else 0, output_padding=1 if i else 0))
            self.instnorms.append(nn.InstanceNorm3d(channels[i+1], affine=True))
            self.resblocks.append(ResBlock(channels[i+1]))
        
        self.final_conv = nn.ConvTranspose3d(64, 1, kernel_size=3, stride=1, padding=1)
        self.final_activation = nn.Sigmoid()

    def forward(self, z):
        for conv, instnorm, resblock in zip(self.convs, self.instnorms, self.resblocks):
            z = conv(z)
            z = instnorm(z)
            z = resblock(z)

        x = self.final_activation(self.final_conv(z))
        
        return x



class VAE(pl.LightningModule):
    def __init__(self, latent_dim=3, max_channels=512, T_max=100, kld_loss_weight=1e-6):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.T_max = T_max
        self.max_channels = max_channels
        self.encoder = Encoder(latent_dim=latent_dim, max_channels=max_channels)
        self.decoder = Decoder(latent_dim=latent_dim, max_channels=max_channels)
        self.kld_loss_weight = kld_loss_weight
        self.automatic_optimization = False

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar
    
    def step(self, batch, batch_idx):
        x, _ = batch
        x_hat, mu, logvar = self.forward(x)

        recon_loss = nn.functional.mse_loss(x_hat, x)
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * self.kld_loss_weight

        total_loss = recon_loss + kld_loss 
        return total_loss, recon_loss, kld_loss

    def training_step(self, batch, batch_idx):
        total_loss, recon_loss, kld_loss = self.step(batch, batch_idx)
        self.log("train_loss", total_loss)
        self.log("train_recon_loss", recon_loss)
        self.log("train_kld_loss", kld_loss)

        vae_opt = self.optimizers()

        vae_opt.zero_grad()
        self.manual_backward(total_loss)
        vae_opt.step()


    def validation_step(self, batch, batch_idx):
        total_loss, recon_loss, kld_loss = self.step(batch, batch_idx)
        self.log("val_loss", total_loss)
        self.log("val_recon_loss", recon_loss)
        self.log("val_kld_loss", kld_loss)


    def configure_optimizers(self):
        vae_opt = torch.optim.Adam(self.parameters(), lr=5e-5) 

        scheduler_vae = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(vae_opt, T_max=self.T_max, eta_min=5e-7),
            'interval': 'epoch',
            'frequency': 1,
            'monitor': 'val_loss'
        }

        return {'optimizer': vae_opt, 'lr_scheduler': scheduler_vae}

    
