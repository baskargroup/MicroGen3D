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
from utils.config_utils import get_model_config, get_context_indices
from utils.dataloader import ImageDataModule
from models.transform import vae_encoder_transform, fp_transform

# === Load Config ===
with open("config_infer.yaml", "r") as f:
    config = yaml.safe_load(f)

# Auto-generate task name
if not config.get('task') or config['task'] == "_":
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    config['task'] = f"run_{timestamp}"

# Prepare directories
data_path = config['data_path']

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Load data
if data_path is not None:
    try:
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
    except Exception as e:
        print(f"Error loading data: {e}")
        train_loader = None
        val_loader = None

# === Load VAE ===
vae_cfg = get_model_config(config, 'vae')
input_shape = tuple(config.get('image_shape', [1, 64, 64, 64]))  # Ensure it's a tuple (C, H, W, D)
print(f"Input shape for VAE: {input_shape}")    
first_layer_downsample=vae_cfg.get('first_layer_downsample', True),
print(f"Using first_layer_downsample={first_layer_downsample} for VAE")
vae = VAE(
    in_shape=input_shape,
    latent_dim=vae_cfg.get('latent_dim_channels', 4),
    max_channels=vae_cfg.get('max_channels', 512),
    first_layer_downsample=vae_cfg.get('first_layer_downsample', True),
    T_max=0, # No training during inference
    kld_loss_weight=vae_cfg['kld_loss_weight']
)

pretrained_path = vae_cfg.get('pretrained_path', '').strip()
try:
    state_dict = torch.load(pretrained_path, map_location=device)
    vae.load_state_dict(state_dict)
    vae = vae.to(device)
    print(f"Loaded pretrained VAE from {pretrained_path}")
except FileNotFoundError:
    print(f"Pretrained VAE weights not found at {pretrained_path}, please check the path.")
    sys.exit(1)
    

# === Load FP ===
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
    T_max=0,  # No training during inference
)

pretrained_path = fp_cfg.get('pretrained_path', '').strip()
try:
    state_dict = torch.load(pretrained_path, map_location=device)
    try:
        fp.load_state_dict(state_dict)  
    except RuntimeError as e:
        print(f"Error loading FP state_dict: {e}")
        print("check one of the two common reasons. switch first_layer_downsample param in vae to opposite value or check the number of attributes in config")
        sys.exit(1)
    fp = fp.to(device)
    print(f"Loaded pretrained FP from {pretrained_path}")
except FileNotFoundError:
    print(f"Pretrained FP weights not found at {pretrained_path}, please check the path.")
    sys.exit(1)

# === Load DDPM ===
ddpm_cfg = get_model_config(config, 'ddpm')
context_indices, context_attributes = get_context_indices(config)
context_dim = len(context_indices)

vae.eval()
fp.eval()

ddpm = DDPM(
    n_T=ddpm_cfg['timesteps'],
    n_feat=ddpm_cfg['n_feat'],
    T_max=0,  # No training during inference
    context_indices=context_indices,
    vae_encoder_transform=vae_encoder_transform(vae),
    fp_transform=fp_transform(fp),
    input_output_channels=vae_cfg['latent_dim_channels']
)

pretrained_path = ddpm_cfg.get('pretrained_path', '').strip()
try:
    state_dict = torch.load(pretrained_path, map_location=device)
    try:
        ddpm.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"Error loading DDPM state_dict: {e}")
        print("check one of the two common reasons. switch first_layer_downsample param in vae to opposite value or check the number of attributes in config")
        sys.exit(1)
    ddpm = ddpm.to(device)
    print(f"Loaded pretrained DDPM from {pretrained_path}")
except FileNotFoundError:
    print(f"Pretrained DDPM weights not found at {pretrained_path}")