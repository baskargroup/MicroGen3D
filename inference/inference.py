import numpy as np
print("loading modules")
import torch
import os
import sys

# Add ../model to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.VAE import VAE
from models.FP import SimpleFC
from models.DDPM import LatentDDPM as DDPM
from training.dataloader import ImageDataModule

import sys
from utils import preprocessors, numpy_exporters
from tqdm import tqdm


data_path  = '../data/sample_test.h5'

batch_size = 20
num_timesteps = 1000
n_feat = 512
learning_rate = 1e-6
max_epochs = 100

image_shape = [1, 64, 64, 64]
attributes = ['ABS_f_D', 'CT_f_D_tort1', 'CT_f_A_tort1']

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

##################################################################################################################################################

sys.exit(0)

DDPM = DDPM(n_T=num_timesteps, n_feat=512, learning_rate=learning_rate, T_max = max_epochs)

# load pretrained DDPM
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = os.listdir('./models/checkpoints')
model_path = os.path.join('./models/checkpoints' , models[0])
checkpoint_ddpm = torch.load(model_path)
DDPM.load_state_dict(checkpoint_ddpm['state_dict'])
DDPM = DDPM.to(device)
print('DDPM Loaded')

# load pretrained VAE
vae = VAE (latent_dim=1)
checkpoint_vae = torch.load ('./vae.ckpt', map_location=device)
vae.load_state_dict(checkpoint_vae['state_dict'])
vae = vae.to(device)
print('vae loaded')

# load pretrained Feature predictor
fc = SimpleFC(input_size=512, output_size=3)
checkpoint_fc = torch.load ('./fp.ckpt', map_location=device)
fc.load_state_dict(checkpoint_fc['state_dict'])
fc = fc.to(device)
print('fc loaded')



params = {}
# original vs vae reconstructed
x = next(iter(val_loader))[0].to(device)
with torch.no_grad():
    # encoding
    mu, logvar = vae.encoder(x)
    z = vae.reparameterize(mu, logvar)
    
    # reconstructing by decoding
    x_hat = vae.decoder(z)
    
    # predicting features
    features = fc(z.flatten(start_dim=1))

    # denoising and generating    
    z_rand = torch.randn(batch_size, 1, 8, 8, 8).to(device)
    print("Denoising image")
    with torch.no_grad():
        z_hat, _ = DDPM.sample_loop(z, features)

    x_generated = vae.decoder(z_hat)

for i in tqdm(range(batch_size)):
    params ['path'] = "/work/mech-ai-scratch/nirmal/generative_model_data/experimental/report/original/"
    params ['file_name'] = str(i)
    params ['arr'] = x[i].squeeze().cpu().numpy()
    exporter = numpy_exporters.ToVti(**params)
    exporter.export()
    
    params ['path'] = "/work/mech-ai-scratch/nirmal/generative_model_data/experimental/report/reconstructed_raw/"
    params ['file_name'] = str(i)
    params ['arr'] = x_hat[i].squeeze().cpu().numpy()
    exporter = numpy_exporters.ToVti(**params)
    exporter.export()
    
    params ['path'] = "/work/mech-ai-scratch/nirmal/generative_model_data/experimental/report/reconstructed_threshold/"
    params ['file_name'] = str(i)
    params ['arr'] = params ['arr'] > 0.5
    exporter = numpy_exporters.ToVti(**params)
    exporter.export()

    params ['path'] = "/work/mech-ai-scratch/nirmal/generative_model_data/experimental/report/generated_raw/"
    params ['file_name'] = str(i)
    params ['arr'] = x_generated[i].squeeze().cpu().numpy()
    exporter = numpy_exporters.ToVti(**params)
    exporter.export()
    
    params ['path'] = "/work/mech-ai-scratch/nirmal/generative_model_data/experimental/report/generated_threshold/"
    params ['file_name'] = str(i)
    params ['arr'] = params ['arr'] > 0.5
    exporter = numpy_exporters.ToVti(**params)
    exporter.export()