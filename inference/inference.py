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
from utils import preprocessors, numpy_exporters

import numpy as np
import pandas as pd
from tqdm import tqdm

# add this import up top with others
from utils.infer_utils import (
    iter_infer_batches,
    run_generation_loop,
)


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
            batch_size=config.get('batch_size', 4),
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
    sys.exit(1)


# ===== Read knobs =====
infer_cfg = config.get("inference", {})
mode = infer_cfg.get("mode", "constant").lower()
print(f"\nInference mode: {mode}\n")
total_samples = int(infer_cfg.get("total_samples", 100))
gen_bs = int(infer_cfg.get("batch_size", 20))

out_cfg = config.get("output", {})
output_dir = out_cfg.get("output_dir", "./output")
write_vti = bool(out_cfg.get("write_vti", True))
write_csv = bool(out_cfg.get("write_csv", True))
threshold = float(out_cfg.get("threshold", 0.5))
save_every_batch = bool(out_cfg.get("save_every_batch", True))

# ===== Shared prep =====
vae.eval(); fp.eval(); ddpm.eval()
context_indices, context_attributes = get_context_indices(config)
attributes_len = len(config.get("attributes", []))
img_shape = tuple(config.get("image_shape", [1, 64, 64, 64]))

print("DDPM conditioning attributes (order):", context_attributes)

# ===== Build batch iterator for the selected mode =====
batch_iter = iter_infer_batches(
    mode,
    total_samples,
    gen_bs,
    vae=vae,
    fp=fp,
    train_loader=locals().get("train_loader"),
    val_loader=locals().get("val_loader"),
    device=device,
    image_shape=img_shape,
    context_indices=context_indices,
    context_attributes=context_attributes,
    attributes_len=attributes_len,
    infer_cfg=infer_cfg,
)

# ===== Run generation & saving =====
run_generation_loop(
    ddpm,
    vae,
    fp,
    batch_iter,
    output_dir=output_dir,
    threshold=threshold,
    write_vti=write_vti,
    write_csv=write_csv,
    save_every_batch=save_every_batch,
    context_attributes=context_attributes,
    full_attributes=config["attributes"],
    numpy_exporters_mod=numpy_exporters,
)
