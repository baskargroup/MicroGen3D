import torch
import torch.nn as nn
import pytorch_lightning as pl


class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(in_channels, affine=True),
            nn.ReLU(),
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(in_channels, affine=True),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))


class Encoder(nn.Module):
    def __init__(self, in_shape, latent_dim=3, max_channels=512, stride1_first_layer=True):
        super().__init__()
        C, H, W, D = in_shape
        self.latent_dim = latent_dim
        self.stride1_first_layer = stride1_first_layer

        layers = []
        in_ch = C
        for out_ch, stride in zip([64, 128, 256, max_channels], 
                                  [1 if stride1_first_layer else 2, 2, 2, 2]):
            layers += [
                nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
                nn.InstanceNorm3d(out_ch, affine=True),
                ResBlock(out_ch),
            ]
            in_ch = out_ch

        layers += [
            nn.Conv3d(in_ch, latent_dim, kernel_size=1),
            nn.InstanceNorm3d(latent_dim, affine=True),
            ResBlock(latent_dim),
        ]

        self.conv_layers = nn.Sequential(*layers)
        self.conv_mu = nn.Conv3d(latent_dim, latent_dim, kernel_size=1)
        self.conv_logvar = nn.Conv3d(latent_dim, latent_dim, kernel_size=1)

    def forward(self, x):
        x = self.conv_layers(x)
        return self.conv_mu(x), self.conv_logvar(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim=3, max_channels=512, stride1_first_layer=True):
        super().__init__()
        channels = [latent_dim, max_channels, 256, 128, 64]
        self.deconv_layers = nn.ModuleList()
        for i in range(len(channels) - 1):
            k, s, p, op = (1, 1, 0, 0) if i == 0 else (3, 2, 1, 1)
            self.deconv_layers.append(nn.Sequential(
                nn.ConvTranspose3d(channels[i], channels[i + 1], kernel_size=k, stride=s, padding=p, output_padding=op),
                nn.InstanceNorm3d(channels[i + 1], affine=True),
                ResBlock(channels[i + 1]),
            ))

        # Final layer (stride adjusted based on encoding)
        self.final_conv = nn.ConvTranspose3d(64, 1, kernel_size=3, stride=2 if not stride1_first_layer else 1, 
                                             padding=1, output_padding=1 if not stride1_first_layer else 0)
        self.final_activation = nn.Sigmoid()

    def forward(self, z):
        for layer in self.deconv_layers:
            z = layer(z)
        return self.final_activation(self.final_conv(z))


class VAE(pl.LightningModule):
    def __init__(self, 
                 in_shape=(1, 64, 64, 64), 
                 latent_dim=3, 
                 max_channels=512, 
                 stride1_first_layer=True,
                 T_max=100, 
                 kld_loss_weight=1e-6):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = Encoder(in_shape, latent_dim, max_channels, stride1_first_layer)
        self.decoder = Decoder(latent_dim, max_channels, stride1_first_layer)
        self.kld_loss_weight = kld_loss_weight
        self.T_max = T_max
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
        x = batch if isinstance(batch, torch.Tensor) else batch[0]
        x_hat, mu, logvar = self.forward(x)
        recon_loss = nn.functional.mse_loss(x_hat, x)
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * self.kld_loss_weight
        return recon_loss + kld_loss, recon_loss, kld_loss

    def training_step(self, batch, batch_idx):
        total_loss, recon_loss, kld_loss = self.step(batch, batch_idx)
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(total_loss)
        opt.step()
        self.log_dict({"train_loss": total_loss, "train_recon": recon_loss, "train_kld": kld_loss})

    def validation_step(self, batch, batch_idx):
        total_loss, recon_loss, kld_loss = self.step(batch, batch_idx)
        self.log_dict({"val_loss": total_loss, "val_recon": recon_loss, "val_kld": kld_loss})

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=5e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.T_max, eta_min=5e-7)
        return {"optimizer": opt, "lr_scheduler": scheduler}
