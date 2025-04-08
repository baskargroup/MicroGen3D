import sys
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import datetime
import yaml
import os

# add ../model to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.VAE import VAE
from models.FP import SimpleFC
from models.DDPM import LatentDDPM as DDPM

from dataloader import ImageDataModule

print("module loaded")

# Load config from YAML
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Auto-generate task name if not provided
if config.get('task') is None:
    time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    config['task'] = 'tmp'
    # config['task'] = f"{time}"

data_path = config['data_path']
model_dir = config['model_dir']
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
transform = None

print('Loading data')
data = ImageDataModule(
    batch_size=config['batch_size'],
    data_path=data_path,
    attributes=config['attributes'],
    transform=transform
)

data.setup()
train_loader = data.train_dataloader()
val_loader = data.val_dataloader()
print('Data loaded')

# training vae
# Logging and Callbacks
wandb_logger = None  # Optional: configure later if needed
checkpoint_callback = ModelCheckpoint(
    monitor='val_recon_loss',
    dirpath=model_dir,
    filename=f"{config['task']}_vae",
    save_top_k=2,
    mode='min',
)

print("Configuration:")
print(config)

vae = VAE(
    latent_dim=config['latent_dim'],
    T_max=config['max_epochs'],
    kld_loss_weight=float(config['kld_loss_weight'])
)

print('Training')
trainer_vae = pl.Trainer(
    max_epochs=config['max_epochs'],
    logger=wandb_logger,
    callbacks=[checkpoint_callback]
)

trainer_vae.fit(vae, train_loader, val_loader)
print('Training vae complete')

# Save the final model
torch.save(vae.state_dict(), os.path.join(model_dir, f"vae_{config['task']}_final_model.pth"))

#################################################################################################
# training FP
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath=model_dir,
    filename=f"{config['task']}_fp",
    save_top_k=2,
    mode='min',
)

num_features = len(config['attributes'])
fp = SimpleFC(
    output_size=num_features,
    vae = vae,
    T_max = config['max_epochs'],
    dropout = 0.1
)

print('Training FP')

trainer_fp = pl.Trainer(max_epochs=config['max_epochs'],
                        logger=wandb_logger,
                        callbacks=[checkpoint_callback])
trainer_fp.fit(fp, train_loader, val_loader)

print('Training FP complete')
# Save the final model
torch.save(fp.state_dict(), os.path.join(model_dir, f"fp_{config['task']}_final_model.pth"))



