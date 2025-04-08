import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import sys

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# vae = VAE (latent_dim=1)
# # load pretrained model
# vae_path = 'vae.ckpt'
# checkpoint = torch.load(vae_path)
# #checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
# vae.load_state_dict(checkpoint['state_dict'])
# vae = vae.to(device)
# print("Loaded VAE model")

class SimpleFC(pl.LightningModule):
    def __init__(self, vae, input_size = 512, output_size = 2, T_max = 100, dropout = 0.1):
        super(SimpleFC, self).__init__()

        self.vae = vae
        self.T_max = T_max
        self.dropout = dropout
        hidden_size1 = input_size // 2
        hidden_size2 = hidden_size1 // 2

        # First hidden layer
        self.fc1 = torch.nn.Linear(input_size, hidden_size1)
        self.norm1 = torch.nn.InstanceNorm1d(hidden_size1)
        self.drop1 = torch.nn.Dropout(self.dropout)
        
        # Second hidden layer
        self.fc2 = torch.nn.Linear(hidden_size1, hidden_size2)
        self.norm2 = torch.nn.InstanceNorm1d(hidden_size2)
        self.drop2 = torch.nn.Dropout(self.dropout)
        
        # Output layer
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
        x_t = self.transform(x)
        y_hat = self(x_t)
        loss = F.mse_loss(y_hat, y)  
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_t = self.transform(x)
        y_hat = self(x_t)
        loss = F.mse_loss(y_hat, y)  
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-5)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.T_max, eta_min=5e-7),
            'interval': 'epoch',  # <- Scheduler steps after each epoch
            'frequency': 1,  # <- Scheduler steps every batch (for batch-level schedulers)
            'monitor': 'val_loss',  # <- Metric to monitor for schedulers with `ReduceLROnPlateau` approach
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}


    def transform(self, x):
        mu, logvar = self.vae.encoder(x)
        z = self.vae.reparameterize(mu, logvar)
        z = z.flatten(start_dim=1)
        return z
