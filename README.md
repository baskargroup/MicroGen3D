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

# Download sample dataset
hf_hub_download(repo_id="BGLab/microgen3D", filename="sample_data.h5", repo_type="dataset", local_dir="data")

# Download experimental pretrained weights
hf_hub_download(repo_id="BGLab/microgen3D", filename="vae.pt", local_dir="models/weights/experimental")
hf_hub_download(repo_id="BGLab/microgen3D", filename="fp.pt", local_dir="models/weights/experimental")
hf_hub_download(repo_id="BGLab/microgen3D", filename="ddpm.pt", local_dir="models/weights/experimental")
```

---

## ⚙️ Configuration

All training and model settings are stored in `config.yaml`.
If `task` is blank or `"_"`, a timestamped task name is generated automatically.

---

### 📄 Full Example `config.yaml`

```yaml
# ================================
# General settings
# ================================
task: "_"                  # Auto-generated if blank or "_"
data_path: "../data/sample_train.h5"  # Path to training dataset
model_dir: "../models/weights/"       # Directory to save model weights
batch_size: 32              # Batch size for training
image_shape: [1, 64, 64, 64]  # Shape of the 3D images [C, D, H, W]
attributes:                 # Full list of attributes predicted by FP
  - ABS_f_D
  - CT_f_D_tort1
  - CT_f_A_tort1

# ================================
# VAE settings
# ================================
vae:
  latent_dim_channels: 1     # Latent space channel size
  kld_loss_weight: 0.000001  # Weight of KL divergence loss
  max_epochs: 1              # Number of training epochs (>=1 means train)
  pretrained_path: "../models/weights/experimental/vae.pt"  # Path to pretrained VAE
  stride1_first_layer: true  # If true, use stride=1 in first conv layer
  max_channels: 512          # Maximum number of channels in VAE

# ================================
# FP settings
# ================================
fp:
  dropout: 0.1               # Dropout probability (0 to 1)
  max_epochs: 2              # Number of training epochs (>=1 means train)
  pretrained_path: "../models/weights/experimental/fp.pt"  # Path to pretrained FP

# ================================
# DDPM settings
# ================================
ddpm:
  timesteps: 1000            # Number of diffusion timesteps
  n_feat: 512                # UNet feature channels (higher = more capacity)
  learning_rate: 0.000001    # Learning rate for optimizer
  max_epochs: 1              # Number of training epochs (>=1 means train)
  pretrained_path: "../models/weights/experimental/ddpm.pt"  # Path to pretrained DDPM
  context_attributes:        # Subset of attributes used as conditioning context
    - ABS_f_D
    - CT_f_D_tort1
    - CT_f_A_tort1

```

---

## 🧪 Pretrained Config Variants

Only change the following fields for each pretrained weights and dataset.
All other parameters can remain the same unless you want to tune them.

---

**🔹 CH 2-Phase**

```yaml
data_path: "../data/sample_CH_two_phase/train/part_*.h5"
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
data_path: "../data/sample_CH_three_phase/train/part_*.h5"
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
data_path: "../data/sample_train.h5"
image_shape: [1, 64, 64, 64]
attributes:
  - ABS_f_D
  - CT_f_D_tort1
  - CT_f_A_tort1
vae.pretrained_path: "../models/weights/experimental/vae.pt"
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
