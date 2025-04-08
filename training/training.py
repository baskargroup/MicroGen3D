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
from dataloader import ImageDataModule

print("module loaded")

# Load config from YAML
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Auto-generate task name if not provided
if config.get('task') is None:
    time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    config['task'] = f"VAE_{time}"

data_path = config['data_path']
model_dir = config['model_dir']
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
transform = None

print('Loading data')
data = ImageDataModule(config['batch_size'], data_path, transform=transform)
data.setup()
train_loader = data.train_dataloader()
val_loader = data.val_dataloader()
print('Data loaded')

# Logging and Callbacks
wandb_logger = None  # Optional: configure later if needed
checkpoint_callback = ModelCheckpoint(
    monitor='val_recon_loss',
    dirpath=model_dir,
    filename=config['task'],
    save_top_k=2,
    mode='min',
)

print("Configuration:")
print(config)

model = VAE(
    latent_dim=config['latent_dim'],
    T_max=config['max_epochs'],
    kld_loss_weight=float(config['kld_loss_weight'])
)

print('Training')
trainer = pl.Trainer(
    max_epochs=config['max_epochs'],
    logger=wandb_logger,
    callbacks=[checkpoint_callback]
)
trainer.fit(model, train_loader, val_loader)
print('Training complete')

# Save the final model
torch.save(model.state_dict(), os.path.join(model_dir, f"{config['task']}_final_model.pth"))
