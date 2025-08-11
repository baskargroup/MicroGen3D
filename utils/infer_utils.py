# utils/infer_utils.py
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# -----------------------------
# Context & latent preparation
# -----------------------------
def make_z_like_from_config(vae, image_shape, batch_size, device):
    """
    Probe the VAE encoder to get the *latent* shape & device.
    Returns a tensor of shape (B, C_lat, D', H', W') on the correct device.
    """
    C, D, H, W = image_shape
    dummy = torch.zeros((1, C, D, H, W), device=device, dtype=torch.float32)
    vae.eval()
    with torch.no_grad():
        mu, _ = vae.encoder(dummy)
    return mu.repeat(batch_size, 1, 1, 1, 1)


def prepare_user_context(user_ctx, context_indices, batch_size, device, attributes_len=None):
    """
    Normalizes user-provided context to shape (B, context_dim), slicing if needed.

    Accepts:
      - 1D: (context_dim,) OR (full_dim,) -> broadcast to (B, context_dim)
      - 2D: (B, context_dim) OR (B, full_dim) -> if full_dim, slice columns

    Args:
      user_ctx: list/tuple/tensor
      context_indices: list[int], indices into full attribute vector
      batch_size: int
      device: torch.device
      attributes_len: optional int to validate full_dim inputs
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
    Returns a (total, context_dim) tensor sampled uniformly per attribute in context_order.
    `ranges_dict` keys MUST match names in `context_order`.
    """
    rows = []
    for k in context_order:
        if k not in ranges_dict:
            raise ValueError(f"Missing random range for context attribute '{k}'")
        lo, hi = ranges_dict[k]
        rows.append(np.random.uniform(lo, hi, total))
    mat = np.stack(rows, axis=1)  # (total, context_dim) in context_order
    return torch.tensor(mat, dtype=torch.float32, device=device)


# -----------------------------
# Batch producer
# -----------------------------
def iter_infer_batches(
    mode,
    total_samples,
    batch_size,
    *,
    vae,
    fp,
    train_loader,
    val_loader,
    device,
    image_shape,
    context_indices,
    context_attributes,
    attributes_len,
    infer_cfg,
):
    """
    Yields tuples: (z_like, context_BxC, optional_x_for_recon)

    mode:
      - "dataset": use dataset to get latent & FP features
      - "constant": broadcast a single context row
      - "random": sample contexts from ranges
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
                z_like = mu[:B]  # latent shape/device
                feats_full = fp(z.flatten(start_dim=1))
                ctx = feats_full[:, context_indices]
            produced += B
            yield z_like, ctx, x  # provide x for recon/quality checks

    elif mode == "constant":
        row = infer_cfg.get("constant_context", [])
        produced = 0
        while produced < total_samples:
            B = min(batch_size, total_samples - produced)
            z_like = make_z_like_from_config(vae, image_shape, B, device)
            ctx = prepare_user_context(row, context_indices, B, device, attributes_len)
            produced += B
            yield z_like, ctx, None

    elif mode == "random":
        rnd = infer_cfg.get("random", {})
        ranges = rnd.get("ranges", {})
        full_ctx_mat = build_random_context_matrix(total_samples, ranges, context_attributes, device)
        produced = 0
        while produced < total_samples:
            B = min(batch_size, total_samples - produced)
            z_like = make_z_like_from_config(vae, image_shape, B, device)
            ctx = full_ctx_mat[produced:produced + B]
            produced += B
            yield z_like, ctx, None

    else:
        raise ValueError(f"Unknown inference.mode '{mode}'. Use 'constant' | 'random' | 'dataset'.")


# -----------------------------
# Generation loop + saving
# -----------------------------
def run_generation_loop(
    ddpm,
    vae,
    fp,
    batch_iter,
    *,
    output_dir,
    threshold,
    write_vti,
    write_csv,
    save_every_batch,
    context_attributes,
    full_attributes,
    numpy_exporters_mod,
):
    """
    Consumes batches yielded by iter_infer_batches, writes VTI(s) and CSV(s).
    """
    os.makedirs(output_dir, exist_ok=True)
    if write_vti:
        for sub in ["generated_raw", "generated_threshold"]:
            os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

    features_list, features_pred_list = [], []
    sample_counter = 0

    for bi, (z_like, ctx, x_opt) in enumerate(tqdm(batch_iter, desc="Infer")):
        B = z_like.size(0)
        with torch.no_grad():
            if x_opt is not None:
                mu, logvar = vae.encoder(x_opt)
                z = vae.reparameterize(mu, logvar)
                x_hat = vae.decoder(z)
            else:
                x_hat = None

            z_gen, _ = ddpm.sample_loop(z_like, ctx)
            x_gen = vae.decoder(z_gen)
            feats_pred = fp(z_gen.flatten(start_dim=1))

        # accumulate for CSV
        features_list.extend(ctx.detach().cpu().numpy())
        features_pred_list.extend(feats_pred.detach().cpu().numpy())

        # VTI writing
        if write_vti:
            for i in range(B):
                base = f"sample_{sample_counter + i:06d}.vti"

                # if x_opt is not None:
                #     for name, arr in {
                #         "original": x_opt[i],
                #         "reconstructed_raw": x_hat[i],
                #         "reconstructed_threshold": (x_hat[i] > threshold).float(),
                #     }.items():
                #         params = {'path': os.path.join(output_dir, name), 'file_name': base,
                #                   'arr': arr.squeeze().detach().cpu().numpy()}
                #         exporter = numpy_exporters_mod.ToVti(**params)
                #         exporter.export()

                for name, arr in {
                    "generated_raw": x_gen[i],
                    "generated_threshold": (x_gen[i] > threshold).float(),
                }.items():
                    params = {'path': os.path.join(output_dir, name), 'file_name': base,
                              'arr': arr.squeeze().detach().cpu().numpy()}
                    exporter = numpy_exporters_mod.ToVti(**params)
                    exporter.export()

        sample_counter += B

        # incremental CSV
        if write_csv and save_every_batch:
            pd.DataFrame(features_list, columns=context_attributes)\
              .to_csv(os.path.join(output_dir, "inputs_context.csv"), index=False)
            pd.DataFrame(features_pred_list, columns=full_attributes)\
              .to_csv(os.path.join(output_dir, "outputs_predicted.csv"), index=False)

    # final CSV
    if write_csv:
        pd.DataFrame(features_list, columns=context_attributes)\
          .to_csv(os.path.join(output_dir, "inputs_context.csv"), index=False)
        pd.DataFrame(features_pred_list, columns=full_attributes)\
          .to_csv(os.path.join(output_dir, "outputs_predicted.csv"), index=False)

    print(f"Done. Wrote {sample_counter} samples to {output_dir}")
