import numpy as np
print("loading modules")
import torch
import os
import sys
import yaml
from tqdm import tqdm

# Add ../model to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.VAE import VAE
from models.FP import SimpleFC
from models.DDPM import LatentDDPM as DDPM
from training.dataloader import ImageDataModule
from utils import preprocessors, numpy_exporters

# Load config
with open('params.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Extract config values
data_path = config['data_path']
batch_size = config['training']['batch_size']
num_timesteps = config['training']['num_timesteps']
learning_rate = config['training']['learning_rate']
max_epochs = config['training']['max_epochs']
n_feat = config['model']['n_feat']
image_shape = config['model']['image_shape']
attributes = config['attributes']
ddpm_path = config['paths']['ddpm_path']
vae_path = config['paths']['vae_path']
fc_path = config['paths']['fc_path']
output_dir = config['paths']['output_dir']

print('Loading data...')
data = ImageDataModule(
    batch_size=batch_size,
    data_path=data_path,
    attributes=attributes,
    image_shape=image_shape,
    transform=None
)
data.setup()
train_loader = data.train_dataloader()
val_loader = data.val_dataloader()
print('Data loaded')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained VAE
vae = VAE(latent_dim=1)
checkpoint_vae = torch.load(vae_path, map_location=device)
vae.load_state_dict(checkpoint_vae['state_dict'])
vae = vae.to(device)
print('VAE loaded')

# Load pretrained feature predictor
dummy_input = torch.randn(1, *image_shape).to(device)
vae.eval()
with torch.no_grad():
    encoded = vae.encoder(dummy_input)
    print(f"Encoded shape: {encoded[0].shape}")
    encoded_shape = encoded[0].flatten(start_dim=1).shape[1]
    print(f"Encoded shape after flattening: {encoded_shape}")

num_features = len(attributes)
fp = SimpleFC(
    input_size=encoded_shape,
    output_size=num_features,
    vae=vae
)
checkpoint_fc = torch.load(fc_path, map_location=device)
fp.load_state_dict(checkpoint_fc['state_dict'])
fp = fp.to(device)
print('Feature predictor loaded')

# Load pretrained DDPM
DDPM = DDPM(vae=vae, fp=fp, n_T=num_timesteps, n_feat=n_feat, learning_rate=learning_rate, T_max=max_epochs)
checkpoint_ddpm = torch.load(ddpm_path, map_location=device)
DDPM.load_state_dict(checkpoint_ddpm['state_dict'])
DDPM = DDPM.to(device)
print('DDPM loaded')


# Run inference
params = {}
x = next(iter(val_loader))[0].to(device)
with torch.no_grad():
    mu, logvar = vae.encoder(x)
    z = vae.reparameterize(mu, logvar)
    x_hat = vae.decoder(z)
    features = fp(z.flatten(start_dim=1))
    z_rand = torch.randn(batch_size, 1, 8, 8, 8).to(device)
    print("Denoising image")
    z_hat, _ = DDPM.sample_loop(z, features)
    x_generated = vae.decoder(z_hat)

# Save outputs
for i in tqdm(range(batch_size)):
    base_name = str(i)

    for name, array in {
        "original": x[i],
        "reconstructed_raw": x_hat[i],
        "reconstructed_threshold": (x_hat[i] > 0.5).float(),
        "generated_raw": x_generated[i],
        "generated_threshold": (x_generated[i] > 0.5).float()
    }.items():
        params = {
            'path': os.path.join(output_dir, name),
            'file_name': base_name,
            'arr': array.squeeze().cpu().numpy()
        }
        exporter = numpy_exporters.ToVti(**params)
        exporter.export()
