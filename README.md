# MicroGen3D

MicroGen3D is a conditional latent diffusion model framework for generating high-resolution 3D multiphase microstructures with user-defined attributes such as volume fraction and tortuosity.  
Designed to accelerate materials discovery, it can synthesize microstructures within a few seconds and predict associated manufacturing parameters.

---

## 🚀 Quick Start

### 🔧 Setup Instructions

```bash
# 1. Clone the repo
git clone https://github.com/baskargroup/MicroGen3D.git
cd MicroGen3D

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Log in to Hugging Face if downloading datasets/weights
huggingface-cli login
````

### 📥 Download Dataset & Pretrained Weights

```python
from huggingface_hub import hf_hub_download
import os

# Download sample dataset
hf_hub_download(
    repo_id="BGLab/microgen3D",
    filename="data/experimental.tar.gz",   # correct remote path
    repo_type="dataset",
    local_dir=""
)

# Download experimental pretrained weights

for fname in ["weights/experimental/vae.pt",
              "weights/experimental/fp.pt",
              "weights/experimental/ddpm.pt"]:
    hf_hub_download(
        repo_id="BGLab/microgen3D",
        filename=fname,                # correct remote path
        repo_type="dataset",
        local_dir=""
    )

```

### 📂 Extract Dataset
```bash 
tar -xzvf data/experimental.tar.gz -C data/ 
``` 

---

## ⚙️ Configuration

All training and model settings are stored in `config_train.yaml`.
If `task` is blank or `"_"`, a timestamped task name is generated automatically.

---

### 📄 Full Example `config_train.yaml`

```yaml
# ================================
# General settings
# ================================
task: "_"                                # str | default="_" | Task name; auto-generated if blank or "_"
data_path: "../data/experimental/sample_train.h5"  # str | REQUIRED | Path or glob pattern to training dataset (e.g., "../data/.../part_*.h5")
model_dir: "../models/weights/"          # str | default="../models/weights/" | Directory where model weights will be saved
batch_size: 32                           # int | default=32 | Number of samples per batch during training
image_shape: [1, 64, 64, 64]              # list[int] | default=[1, 64, 64, 64] | Shape of 3D input [C, D, H, W]
attributes:                               # list[str] | REQUIRED | Full list of attributes predicted by FP
  - ABS_f_D
  - CT_f_D_tort1
  - CT_f_A_tort1

# ================================
# VAE settings
# ================================
vae:
  latent_dim_channels: 1                  # int | default=1 | Latent space channel size
  kld_loss_weight: 0.000001               # float | default=1e-6 | Weight of KL divergence loss term
  max_epochs: 1                           # int | default=0 | Number of epochs to train (>=1 = train, 0 = skip training)
  pretrained_path: "../weights/experimental/vae.pt"  # str | default="" | Path to pretrained VAE weights (empty or wrong or null path = train from scratch)
  first_layer_downsample: true            # bool | default=False | If true, first conv layer downsamples input (stride=2), else uses stride=1
  max_channels: 512                       # int | default=512 | Max number of feature channels in VAE encoder/decoder

# ================================
# FP settings
# ================================
fp:
  dropout: 0.1                             # float in [0,1] | default=0.1 | Dropout probability for fully connected layers
  max_epochs: 2                            # int | default=0 | Number of epochs to train (>=1 = train, 0 = skip training)
  pretrained_path: "../weights/experimental/fp.pt"   # str | default="" | Path to pretrained FP weights (empty or wrong or null path = train from scratch)

# ================================
# DDPM settings
# ================================
ddpm:
  timesteps: 1000                          # int | default=1000 | Number of diffusion timesteps
  n_feat: 512                              # int | default=512 | UNet base feature channels (higher = more capacity)
  learning_rate: 0.000001                  # float | default=1e-6 | Learning rate for optimizer
  max_epochs: 1                            # int | default=0 | Number of epochs to train (>=1 = train, 0 = skip training)
  pretrained_path: "../weights/experimental/ddpm.pt" # str | default="" | Path to pretrained DDPM weights (empty or wrong or null path = train from scratch)
  context_attributes:                      # list[str] | default=<attributes> | Subset of `attributes` used as DDPM conditioning
    - ABS_f_D
    - CT_f_D_tort1
    - CT_f_A_tort1

```

---

## 🧪 Pretrained Config Variants

To use other pretrained weights or datasets, you can copy the example configurations below into your `config_train.yaml` file. But before doing so, ensure you have downloaded the corresponding pretrained weights and dataset files as described earlier.

Only change the following fields for each pretrained weights and dataset. All other parameters can remain the same unless you want to tune them.

---

**⚠️ Important — Required Parameters for Pretrained Weights**
The following parameters **must match exactly** when using the provided pretrained weights.
If you change any of these values, the pretrained models will not load or function properly.

* `data_path`
* `image_shape`
* `attributes`
* `vae.pretrained_path`
* `fp.pretrained_path`
* `ddpm.pretrained_path`
* `ddpm.context_attributes`

All other parameters (e.g., batch size, learning rate, max\_epochs) can be adjusted for fine-tuning if desired.


**🔹 CH 2-Phase**

```yaml
data_path: "../data/sample_CH_two_phase/train/part_*.h5" # wildcard to use all parts. If you want to use only one part, change it to "../data/sample_CH_two_phase/train/part_1.h5" etc.
image_shape: [1, 128, 128, 64]
attributes:
  - norm_STAT_e
  - ABS_wf_D
  - ABS_f_D
  - DISS_wf10_D
  - CT_f_e_conn
  - CT_f_D_tort1
  - CT_f_A_tort1
vae.pretrained_path: "../models/weights/CH_2phase/vae.pt"
vae.latent_dim_channels: 4
fp.pretrained_path:  "../models/weights/CH_2phase/fp.pt"
ddpm.pretrained_path: "../models/weights/CH_2phase/ddpm.pt"
ddpm.context_attributes:
  - ABS_f_D
  - CT_f_D_tort1
  - CT_f_A_tort1
```

---

**🔹 CH 3-Phase**

```yaml
data_path: "../data/sample_CH_three_phase/train/part_*.h5" # wildcard to use all parts. If you want to use only one part, change it to "../data/sample_CH_three_phase/train/part_1.h5" etc.
image_shape: [1, 128, 128, 64]
attributes:
  - vol_frac_D
  - vol_frac_M
  - tortuosity_A
  - tortuosity_D
  - phi
  - chi
  - log_time
vae.pretrained_path: "../models/weights/CH_3phase/vae.pt"
vae.latent_dim_channels: 4
fp.pretrained_path:  "../models/weights/CH_3phase/fp.pt"
ddpm.pretrained_path: "../models/weights/CH_3phase/ddpm.pt"
ddpm.context_attributes:
  - vol_frac_D
  - vol_frac_M
  - tortuosity_A
  - tortuosity_D
```

---

**🔹 Experimental**

```yaml
data_path: "../data/experimental/train/train.h5"  # Path to the experimental dataset
image_shape: [1, 64, 64, 64]
attributes:
  - ABS_f_D
  - CT_f_D_tort1
  - CT_f_A_tort1
vae.pretrained_path: "../models/weights/experimental/vae.pt"
vae.latent_dim_channels: 1
fp.pretrained_path:  "../models/weights/experimental/fp.pt"
ddpm.pretrained_path: "../models/weights/experimental/ddpm.pt"
ddpm.context_attributes:
  - ABS_f_D
  - CT_f_D_tort1
  - CT_f_A_tort1
```

---

## 🏋️ Training

From the repo root:

```bash
cd training
python training.py
```

When training starts, you will see a **Training Strategy Summary** showing for each model (VAE, FP, DDPM) whether pretrained weights were loaded and how many epochs it will train.

---

## 📄 License

[MIT License](LICENSE)
