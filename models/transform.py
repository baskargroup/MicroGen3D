import torch

def vae_encoder_transform(vae):
    def transform(x):
        with torch.no_grad():  # Ensure VAE doesn't track gradients
            mu, logvar = vae.encoder(x)
            z = vae.reparameterize(mu, logvar)
        return z
    return transform

def fp_transform(fp):
    def transform(z):
        with torch.no_grad():  # Ensure FP doesn't track gradients
            features = fp(z.flatten(start_dim=1))
        return features
    return transform