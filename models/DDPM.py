import torch, os
import torch.nn.functional as F
import pytorch_lightning as pl
from tqdm import tqdm
from einops import rearrange
from torchvision.utils import save_image, make_grid
import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './')))

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# vae_path = 'vae.ckpt'
# vae = VAE (latent_dim=1)
# checkpoint_vae = torch.load (vae_path, map_location=device)
# vae.load_state_dict(checkpoint_vae['state_dict'])
# vae = vae.to(device)
# print('VAE loaded')

# fc_path = 'fp.ckpt'
# num_features = len(dataloader.attributes)
# fc = SimpleFC(input_size=512, output_size=num_features)
# checkpoint_fc = torch.load (fc_path, map_location=device)
# fc.load_state_dict(checkpoint_fc['state_dict'])
# fc = fc.to(device)
# print('FP loaded')


import torch
from torch import nn
from pytorch_lightning import LightningModule, Trainer

from .unet import ContextUnet_3D_2lvls as UNet
from math import pi

# Define the DDPM
def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 1e-4
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float32)

class LatentDDPM(pl.LightningModule):
    def __init__(self, vae, fp, n_T=1000, n_feat=128, learning_rate=1e-5, T_max = 500, context_dim=3):
        super(LatentDDPM, self).__init__()

        self.vae = vae
        self.fp = fp

        self.nn_model = UNet(in_channels=1, out_channels=1, n_feat=n_feat, context_dim=context_dim)

        # setting up the diffusion part
        self.betas = linear_beta_schedule(n_T)
        self.ddpm_schedules = self.register_ddpm_schedules(self.betas)
        self.n_T = n_T
        self.learning_rate = learning_rate
        self.T_max = T_max

        print(f'LatentDDPM initialized with n_T={self.n_T}, learning_rate={self.learning_rate}, T_max={self.T_max}, n_feat={n_feat}')
        for k, v in self.ddpm_schedules.items():
            self.register_buffer(k, v)

        
    def register_ddpm_schedules(self, beta_t):
        """
        Returns pre-computed schedules for DDPM sampling, training process.
        """

        sqrt_beta_t = torch.sqrt(beta_t)
        alpha_t = 1 - beta_t
        log_alpha_t = torch.log(alpha_t)
        alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

        sqrtab = torch.sqrt(alphabar_t)
        oneover_sqrta = 1 / torch.sqrt(alpha_t)

        sqrtmab = torch.sqrt(1 - alphabar_t)
        mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

        return {
            "alpha_t": alpha_t,  # \alpha_t
            "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
            "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
            "alphabar_t": alphabar_t,  # \bar{\alpha_t}
            "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
            "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
            "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
        }

    def diffusion(self, x, _ts, noise):
        x_t = (
            self.sqrtab[_ts.long(), None, None, None, None] * x
            + self.sqrtmab[_ts.long(), None, None, None, None] * noise
        )
        return x_t

    def forward(self, x):
        noise = torch.randn_like(x)  # eps ~ N(0, 1)
        _ts = torch.randint(1, self.n_T, (x.shape[0],)).to(x).long()  # t ~ Uniform(0, n_T)
        x_t = self.diffusion(x, _ts, noise)
        return F.mse_loss(noise, self.nn_model(x_t, _ts / self.n_T, self.context))

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = self._transform(x)
        loss = self.forward(x)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x = self._transform(x)
        loss = self.forward(x)
        self.log("val_loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        eta_min = self.learning_rate * 0.1
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.T_max, eta_min=eta_min),
            'interval': 'epoch',  # <- Scheduler steps after each epoch
            'frequency': 1,  # <- Scheduler steps every batch (for batch-level schedulers)
            'monitor': 'val_loss',  # <- Metric to monitor for schedulers with `ReduceLROnPlateau` approach
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def _transform(self,images):
        with torch.no_grad():
            mu, logvar = self.vae.encoder(images)
            z = self.vae.reparameterize(mu, logvar)
            # in this run volume fraction is the context
            features = self.fp(z.flatten(start_dim=1))
            features = features[:,:3]
            self.context = features 
        return z

    def generate(self, z, num_timesteps, context):
        """Generate a new sample starting from noise `z`."""
        for timestep in reversed(range(num_timesteps)):
            z = self.nn_model(z, (torch.tensor(timestep) / self.n_T).to(z), context)
        return z
    
    def on_train_epoch_end(self):
        # print current scheduled learning rate
        print('current learning rate', self.trainer.optimizers[0].param_groups[0]['lr'])

    def sample_loop(self, batch, context):
        x_i = torch.randn_like(batch).to(batch)  # x_T ~ N(0, 1), sample initial noise
        anim = []
        for i in range(self.n_T - 2, 0, -1):
            t_is = torch.tensor([i / self.n_T]).to(batch)
            t_is = t_is.repeat(batch.size(0),1,1,1,1)

            z = torch.randn_like(batch).to(batch) if i > 1 else 0

            eps = self.nn_model(x_i, t_is, context)
            x_i = x_i[:batch.size(0)]
            x_i = (
                    self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                    + self.sqrt_beta_t[i] * z
            )

            if i % 10 == 0:
                anim.append(x_i)
        return x_i, anim
