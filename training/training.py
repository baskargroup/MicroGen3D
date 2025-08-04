import sys
import os
import yaml
import datetime
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import time

# Add ../model to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.VAE import VAE
from models.FP import SimpleFC
from models.DDPM import LatentDDPM as DDPM
from dataloader import ImageDataModule
from models.transform import vae_encoder_transform, fp_transform

from config_utils import get_model_config

# === Load Config ===
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Auto-generate task name
if not config.get('task') or config['task'] == "_":
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    config['task'] = f"run_{timestamp}"

# Prepare directories
data_path = config['data_path']
model_dir = config['model_dir']
os.makedirs(model_dir, exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Summary
print("\n=== TRAINING STRATEGY SUMMARY ===")
for model_name in ['vae', 'fp', 'ddpm']:
    cfg = get_model_config(config, model_name)
    pretrained = os.path.isfile(cfg['pretrained_path'])
    if pretrained:
        print(f"{model_name.upper()}: Load pretrained weights from '{cfg['pretrained_path']}' and train for {cfg['max_epochs']} epoch(s).")
    else:
        print(f"{model_name.upper()}: No pretrained weights found. Train from scratch for {cfg['max_epochs']} epoch(s).")
print("=================================\n")

time.sleep(3)  # Pause for better readability

# Load data
print('Loading data...')
data = ImageDataModule(
    batch_size=config.get('batch_size', 32),
    data_path=data_path,
    attributes=config.get('attributes', []),
    image_shape=config.get('image_shape', [1, 64, 64, 64]),
    transform=None,
)
data.setup()
train_loader = data.train_dataloader()
val_loader = data.val_dataloader()
print('Data loaded.')

# Checkpoint helper
def checkpoint_dir(name, monitor, mode='min'):
    return ModelCheckpoint(
        monitor=monitor,
        dirpath=model_dir,
        filename=f"{config['task']}_{name}",
        save_top_k=1,
        mode=mode
    )

wandb_logger = None  # Optional

# === Train VAE ===
vae_cfg = get_model_config(config, 'vae')
vae = VAE(
    latent_dim_channels=vae_cfg['latent_dim_channels'],
    T_max=vae_cfg['max_epochs'],
    kld_loss_weight=vae_cfg['kld_loss_weight']
)
if os.path.isfile(vae_cfg['pretrained_path']):
    state_dict = torch.load(vae_cfg['pretrained_path'], map_location=device)
    vae.load_state_dict(state_dict)
    print(f"Loaded pretrained VAE from {vae_cfg['pretrained_path']}")
vae = vae.to(device)

if vae_cfg['max_epochs'] >= 1:
    print('Training VAE...')
    trainer = pl.Trainer(
        max_epochs=vae_cfg['max_epochs'],
        logger=wandb_logger,
        callbacks=[checkpoint_dir('vae', 'val_recon_loss')],
        accelerator="auto",
        devices=1
    )
    trainer.fit(vae, train_loader, val_loader)
    torch.save(vae.state_dict(), os.path.join(model_dir, f"vae_{config['task']}_final_model.pth"))
    print('VAE training complete.')

vae = vae.to(device)

# === Train FP ===
dummy_input = torch.randn(1, *config['image_shape']).to(device)
vae.eval()
with torch.no_grad():
    encoded = vae.encoder(dummy_input)
    print(f"Encoded shape: {encoded[0].shape}, Logvar shape: {encoded[1].shape}")
    encoded_shape = encoded[0].flatten(start_dim=1).shape[1]

fp_cfg = get_model_config(config, 'fp')
num_features = len(config.get('attributes', []))

fp = SimpleFC(
    input_size=encoded_shape,
    output_size=num_features,
    vae_encoder_transform=vae_encoder_transform(vae),
    T_max=fp_cfg['max_epochs'],
    dropout=fp_cfg['dropout']
)
if os.path.isfile(fp_cfg['pretrained_path']):
    state_dict = torch.load(fp_cfg['pretrained_path'], map_location=device)
    fp.load_state_dict(state_dict)
    print(f"Loaded pretrained FP from {fp_cfg['pretrained_path']}")
fp = fp.to(device)

if fp_cfg['max_epochs'] >= 1:
    print('Training FP...')
    trainer = pl.Trainer(
        max_epochs=fp_cfg['max_epochs'],
        logger=wandb_logger,
        callbacks=[checkpoint_dir('fp', 'val_loss')],
        accelerator="auto",
        devices=1
    )
    trainer.fit(fp, train_loader, val_loader)
    torch.save(fp.state_dict(), os.path.join(model_dir, f"fp_{config['task']}_final_model.pth"))
    print('FP training complete.')

fp = fp.to(device)

# === Train DDPM ===
ddpm_cfg = get_model_config(config, 'ddpm')
vae.eval()
fp.eval()

ddpm = DDPM(
    n_T=ddpm_cfg['timesteps'],
    n_feat=ddpm_cfg['n_feat'],
    learning_rate=ddpm_cfg['learning_rate'],
    T_max=ddpm_cfg['max_epochs'],
    context_dim=num_features,
    vae_encoder_transform=vae_encoder_transform(vae),
    fp_transform=fp_transform(fp),
    input_output_channels=vae_cfg['latent_dim_channels']
)

if os.path.isfile(ddpm_cfg['pretrained_path']):
    state_dict = torch.load(ddpm_cfg['pretrained_path'], map_location=device)
    ddpm.load_state_dict(state_dict)
    print(f"Loaded pretrained DDPM from {ddpm_cfg['pretrained_path']}")
ddpm = ddpm.to(device)

if ddpm_cfg['max_epochs'] >= 1:
    print('Training DDPM...')
    trainer = pl.Trainer(
        max_epochs=ddpm_cfg['max_epochs'],
        logger=wandb_logger,
        callbacks=[checkpoint_dir('ddpm', 'val_loss')],
        accelerator="auto",
        devices=1
    )
    trainer.fit(ddpm, train_loader, val_loader)
    torch.save(ddpm.state_dict(), os.path.join(model_dir, f"ddpm_{config['task']}_final_model.pth"))
    print('DDPM training complete.')

print('Training pipeline complete.')
