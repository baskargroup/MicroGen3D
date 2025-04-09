import sys
import os
import yaml
import datetime
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

# Add ../model to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.VAE import VAE
from models.FP import SimpleFC
from models.DDPM import LatentDDPM as DDPM
from dataloader import ImageDataModule

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Auto-generate task name if not provided
if not config.get('task'):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    config['task'] = f"run_{timestamp}"

# Prepare directories
data_path = config['data_path']
model_dir = config['model_dir']
os.makedirs(model_dir, exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
print('Loading data...')
data = ImageDataModule(
    batch_size=config['batch_size'],
    data_path=data_path,
    attributes=config['attributes'],
    image_shape=config['image_shape'],
    transform=None,
)
data.setup()
train_loader = data.train_dataloader()
val_loader = data.val_dataloader()
print('Data loaded.')

# Callbacks helper
def checkpoint_dir(name, monitor, mode='min'):
    return ModelCheckpoint(
        monitor=monitor,
        dirpath=model_dir,
        filename=f"{config['task']}_{name}",
        save_top_k=2,
        mode=mode
    )

wandb_logger = None  # Optional: configure later

print("Configuration:")
print(config)

#################################
# Train VAE
vae_cfg = config['vae']
vae = VAE(
    latent_dim_channels=vae_cfg['latent_dim_channels'],
    T_max=vae_cfg['max_epochs'],
    kld_loss_weight=float(vae_cfg['kld_loss_weight'])
)

if vae_cfg.get('pretrained', False):
    assert os.path.isfile(vae_cfg['pretrained_path']), f"VAE pretrained model not found at {vae_cfg['pretrained_path']}"
    vae.load_state_dict(torch.load(vae_cfg['pretrained_path'], map_location=device))
    print(f"Loaded pretrained VAE from {vae_cfg['pretrained_path']}")
else:
    vae = vae.to(device)
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

#################################
# Train FP
dummy_input = torch.randn(1, *config['image_shape']).to(device)
vae.eval()
with torch.no_grad():
    encoded = vae.encoder(dummy_input)
    print(f"Encoded shape: {encoded[0].shape}")
    encoded_shape = encoded[0].flatten(start_dim=1).shape[1]
    print(f"Encoded shape after flattening: {encoded_shape}")

fp_cfg = config['fp']
num_features = len(config['attributes'])
fp = SimpleFC(
    input_size=encoded_shape,
    output_size=num_features,
    vae=vae,
    T_max=fp_cfg['max_epochs'],
    dropout=fp_cfg.get('dropout', 0.1)
)

if fp_cfg.get('pretrained', False):
    assert os.path.isfile(fp_cfg['pretrained_path']), f"FP pretrained model not found at {fp_cfg['pretrained_path']}"
    fp.load_state_dict(torch.load(fp_cfg['pretrained_path'], map_location=device))
    print(f"Loaded pretrained FP from {fp_cfg['pretrained_path']}")
else:
    fp = fp.to(device)
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

#################################
# Train DDPM
ddpm_cfg = config['ddpm']
fp.eval()
vae.eval()
ddpm = DDPM(
    n_T=ddpm_cfg['timesteps'],
    n_feat=ddpm_cfg['n_feat'],
    learning_rate=float(ddpm_cfg['learning_rate']),
    T_max=ddpm_cfg['max_epochs'],
    context_dim=num_features,
    vae=vae,
    fp=fp,
    input_output_channels=vae_cfg['latent_dim_channels']
)
ddpm = ddpm.to(device)

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

print('Training pipeline complete.')
