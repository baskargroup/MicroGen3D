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

import numpy as np
import pandas as pd
from tqdm import tqdm

# ===== Helpers =====
def make_z_like_from_config(vae, image_shape, batch_size, device):
    C, D, H, W = image_shape
    dummy = torch.zeros((1, C, D, H, W), device=device, dtype=torch.float32)
    vae.eval()
    with torch.no_grad():
        mu, _ = vae.encoder(dummy)
    # mu is (1, C_lat, D', H', W'); repeat to desired batch size
    return mu.repeat(batch_size, 1, 1, 1, 1)

def prepare_user_context(user_ctx, context_indices, batch_size, device, attributes_len=None):
    """
    Accepts:
      - (context_dim,) or (full_dim,) -> broadcast to (B, context_dim)
      - (B, context_dim) or (B, full_dim) -> if full_dim, slices columns
    """
    if isinstance(user_ctx, (list, tuple)):
        user_ctx = torch.tensor(user_ctx, dtype=torch.float32)
    if not torch.is_tensor(user_ctx):
        raise ValueError("context must be a tensor/list/tuple")

    ctx_dim = len(context_indices)
    user_ctx = user_ctx.to(device).float()

    if user_ctx.ndim == 1:
        D = user_ctx.numel()
        if D == ctx_dim:
            ctx = user_ctx.unsqueeze(0).repeat(batch_size, 1)
        else:
            if attributes_len is not None and D != attributes_len:
                raise ValueError(f"1D context length {D} != expected attributes length {attributes_len}")
            ctx = user_ctx[context_indices].unsqueeze(0).repeat(batch_size, 1)
    elif user_ctx.ndim == 2:
        B_in, D = user_ctx.shape
        if B_in == 1 and batch_size > 1:
            user_ctx = user_ctx.repeat(batch_size, 1)
            B_in = batch_size
        if B_in != batch_size:
            raise ValueError(f"context batch {B_in} != requested batch_size {batch_size}")
        if D == ctx_dim:
            ctx = user_ctx
        else:
            if attributes_len is not None and D != attributes_len:
                raise ValueError(f"2D context width {D} != expected attributes length {attributes_len}")
            ctx = user_ctx[:, context_indices]
    else:
        raise ValueError(f"context must be 1D or 2D; got ndim={user_ctx.ndim}")

    return ctx  # (B, ctx_dim)

def build_random_context_matrix(total, ranges_dict, context_order, device):
    """
    Returns (total, context_dim) sampled uniformly per key in context_order.
    ranges_dict keys must match names in context_order.
    """
    rows = []
    for k in context_order:
        if k not in ranges_dict:
            raise ValueError(f"Missing random range for context attribute '{k}'")
        lo, hi = ranges_dict[k]
        rows.append(np.random.uniform(lo, hi, total))
    mat = np.stack(rows, axis=1)  # (total, context_dim) in context_order
    return torch.tensor(mat, dtype=torch.float32, device=device)

# ===== Read knobs =====
infer_cfg = config.get("inference", {})
mode = infer_cfg.get("mode", "constant").lower()
print(f"\nInference mode: {mode} \n")
total_samples = int(infer_cfg.get("total_samples", 100))
gen_bs = int(infer_cfg.get("batch_size", 20))

out_cfg = config.get("output", {})
output_dir = out_cfg.get("output_dir", "./output")
write_vti = bool(out_cfg.get("write_vti", True))
write_csv = bool(out_cfg.get("write_csv", True))
threshold = float(out_cfg.get("threshold", 0.5))
save_every_batch = bool(out_cfg.get("save_every_batch", True))
csv_inputs = out_cfg.get("csv_inputs", "inputs_context.csv")
csv_outputs = out_cfg.get("csv_outputs", "outputs_predicted.csv")

os.makedirs(output_dir, exist_ok=True)
if write_vti:
    for sub in ["generated_raw", "generated_threshold"]:
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

# ===== Shared prep =====
vae.eval(); fp.eval(); ddpm.eval()
context_indices, context_attributes = get_context_indices(config)
attributes_len = len(config.get("attributes", []))
img_shape = tuple(config.get("image_shape", [1, 64, 64, 64]))
io_channels = vae_cfg['latent_dim_channels']

print("DDPM conditioning attributes (order):", context_attributes)

# For CSV outputs
features_list, features_pred_list = [], []

# ===== Batch generator depending on mode =====
def batches():
    """
    Yields tuples: (z_like, context_BxC, optional_x_for_recon) where
      - z_like defines the latent shape/device (content ignored in sampling)
      - context_BxC has shape (B, context_dim)
      - optional_x_for_recon is None for constant/random modes
    """
    if mode == "dataset":
        which = infer_cfg.get("dataset_loader", "val").lower()
        loader = val_loader if which == "val" else train_loader
        if loader is None:
            raise RuntimeError("Dataset mode selected but data_path/loader is not available.")
        produced = 0
        for x, _ in loader:
            if produced >= total_samples:
                break
            x = x.to(device)
            B = min(x.size(0), total_samples - produced)
            x = x[:B]
            with torch.no_grad():
                mu, logvar = vae.encoder(x)
                z = vae.reparameterize(mu, logvar)
                z_like = mu[:B]  # correct latent shape/device
                feats_full = fp(z.flatten(start_dim=1))
                ctx = feats_full[:, context_indices]
            produced += B
            yield z_like, ctx, x  # x provided for recon/qual checks

    elif mode == "constant":
        # one row (constant_context) broadcast to all samples
        row = infer_cfg.get("constant_context", [])
        produced = 0
        while produced < total_samples:
            B = min(gen_bs, total_samples - produced)
            z_like = make_z_like_from_config(vae, img_shape, B, device)
            ctx = prepare_user_context(row, context_indices, B, device, attributes_len)
            produced += B
            yield z_like, ctx, None

    elif mode == "random":
        rnd = infer_cfg.get("random", {})
        ranges = rnd.get("ranges", {})
        # build full matrix for total_samples in context order
        full_ctx_mat = build_random_context_matrix(total_samples, ranges, context_attributes, device)
        produced = 0
        while produced < total_samples:
            B = min(gen_bs, total_samples - produced)
            z_like = make_z_like_from_config(vae, img_shape, B, device)
            ctx = full_ctx_mat[produced:produced+B]
            produced += B
            yield z_like, ctx, None
    else:
        raise ValueError(f"Unknown inference.mode '{mode}'. Use 'constant' | 'random' | 'dataset'.")

# ===== Run generation =====
sample_counter = 0
for bi, (z_like, ctx, x_opt) in enumerate(tqdm(batches(), desc=f"Infer[{mode}]")):
    B = z_like.size(0)
    assert z_like.size(1) == io_channels, f"Latent channels {z_like.size(1)} != DDPM expected {io_channels}"

    with torch.no_grad():
        # Optional recon if dataset mode
        if x_opt is not None:
            mu, logvar = vae.encoder(x_opt)
            z = vae.reparameterize(mu, logvar)
            x_hat = vae.decoder(z)
        else:
            x_hat = None

        # Generate
        z_gen, _ = ddpm.sample_loop(z_like, ctx)
        x_gen = vae.decoder(z_gen)

        # Predict features from generated latents
        feats_pred = fp(z_gen.flatten(start_dim=1))

    # Accumulate CSV data
    features_list.extend(ctx.detach().cpu().numpy())                 # inputs (context)
    features_pred_list.extend(feats_pred.detach().cpu().numpy())     # predicted from gen

    # Save VTI grids if requested
    if write_vti:
        for i in range(B):
            base = f"sample_{sample_counter + i:06d}.vti"
            if x_opt is not None:
                for name, arr in {
                    "original": x_opt[i],
                    "reconstructed_raw": x_hat[i],
                    "reconstructed_threshold": (x_hat[i] > threshold).float(),
                }.items():
                    params = {'path': os.path.join(output_dir, name), 'file_name': base, 'arr': arr.squeeze().detach().cpu().numpy()}
                    exporter = numpy_exporters.ToVti(**params); exporter.export()

            for name, arr in {
                "generated_raw": x_gen[i],
                "generated_threshold": (x_gen[i] > threshold).float(),
            }.items():
                params = {'path': os.path.join(output_dir, name), 'file_name': base, 'arr': arr.squeeze().detach().cpu().numpy()}
                exporter = numpy_exporters.ToVti(**params); exporter.export()

    sample_counter += B

    # Incremental CSV saving if desired
    if write_csv and save_every_batch:
        pd.DataFrame(features_list, columns=context_attributes).to_csv(os.path.join(output_dir, csv_inputs), index=False)
        pd.DataFrame(features_pred_list, columns=config['attributes']).to_csv(os.path.join(output_dir, csv_outputs), index=False)

# Final CSV dump
if write_csv:
    pd.DataFrame(features_list, columns=context_attributes).to_csv(os.path.join(output_dir, csv_inputs), index=False)
    pd.DataFrame(features_pred_list, columns=config['attributes']).to_csv(os.path.join(output_dir, csv_outputs), index=False)

print(f"Done. Wrote {sample_counter} samples to {output_dir}")



# # ---- Inference (validation) ----
# vae.eval(); fp.eval(); ddpm.eval()

# # how many val batches to generate from
# num_batches = 2
# # <- set as you like
# output_dir = config.get('output_dir', 'output')
# os.makedirs(output_dir, exist_ok=True)

# with torch.no_grad():
#     for n, (x, _) in enumerate(train_loader): # no need to use val_loader since we should use the validation dataset anyway
#         if n >= num_batches:
#             break

#         x = x.to(device)
#         B = x.size(0)

#         # --- Get latent shape (we only need shape/device for sampling) ---
#         # Use encoder to get latent shape; contents are NOT used by sample_loop.
#         mu, logvar = vae.encoder(x)
#         z_like = mu  # shape: (B, C_latent, D', H', W'), correct device

#         # --- Reconstruct (optional sanity check) ---
#         z = vae.reparameterize(mu, logvar)
#         x_hat = vae.decoder(z)

#         # --- Build context from FP and SLICE to DDPM context_indices ---
#         # FP expects flattened latent; use the same path as training.
#         features_full = fp(z.flatten(start_dim=1))            # shape: (B, len(attributes))
#         context = features_full[:, context_indices]           # shape: (B, len(context_indices))

#         # --- Generate from noise in latent space using context ---
#         print("Denoising latent...")
#         z_gen, _ = ddpm.sample_loop(z_like, context)          # starts from noise; uses z_like.shape/device

#         # --- Decode generated latent to image space ---
#         x_gen = vae.decoder(z_gen)

#         # --- Save outputs ---
#         for i in range(B):
#             base_name = f"batch_{n}_sample_{i}.vti"
#             for name, array in {
#                 "original": x[i],
#                 "reconstructed_raw": x_hat[i],
#                 "reconstructed_threshold": (x_hat[i] > 0.5).float(),
#                 "generated_raw": x_gen[i],
#                 "generated_threshold": (x_gen[i] > 0.5).float()
#             }.items():
#                 params = {
#                     'path': os.path.join(output_dir, name),
#                     'file_name': base_name,
#                     'arr': array.squeeze().detach().cpu().numpy()
#                 }
#                 exporter = numpy_exporters.ToVti(**params)
#                 exporter.export()
