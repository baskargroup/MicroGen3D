import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import sys

class SimpleFC(pl.LightningModule):
    def __init__(self, vae_encoder_transform, input_size=512, output_size=2, T_max=100, dropout=0.1):
        super(SimpleFC, self).__init__()

        self.vae_encoder_transform = vae_encoder_transform
        self.T_max = T_max
        self.dropout = dropout

        hidden_size1 = input_size // 2
        hidden_size2 = hidden_size1 // 2

        self.fc1 = torch.nn.Linear(input_size, hidden_size1)
        self.norm1 = torch.nn.InstanceNorm1d(hidden_size1)
        self.drop1 = torch.nn.Dropout(self.dropout)

        self.fc2 = torch.nn.Linear(hidden_size1, hidden_size2)
        self.norm2 = torch.nn.InstanceNorm1d(hidden_size2)
        self.drop2 = torch.nn.Dropout(self.dropout)

        self.fc3 = torch.nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.norm1(x)
        x = self.drop1(x)

        x = F.relu(self.fc2(x))
        x = self.norm2(x)
        x = self.drop2(x)

        x = self.fc3(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_t = self.vae_encoder_transform(x)  # Call external transform
        y_hat = self(x_t.flatten(start_dim=1))
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_t = self.vae_encoder_transform(x)
        y_hat = self(x_t.flatten(start_dim=1))
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-5)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.T_max, eta_min=5e-7),
            'interval': 'epoch',
            'frequency': 1,
            'monitor': 'val_loss',
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
