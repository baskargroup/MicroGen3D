# MicroGen3D

MicroGen3D is a conditional latent diffusion model framework for generating high-resolution, 3D multiphase microstructures based on user-defined attributes such as volume fraction and tortuosity. Designed to accelerate materials discovery, it can synthesize 3D microstructures within seconds and predict their corresponding manufacturing parameters.

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
huggingface-cli login # enter your token when prompted
````

### 📥 Download Dataset & Pretrained Weights
Copy and run this code in a Python script or notebook to download the sample dataset and pretrained weights into the current directory with the correct folder structure. For more details about the dataset and pretrained weights, visit the Hugging Face page.

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
## 🏋️ Training

From the repo root:

```bash
cd training
python training.py
```

All training settings are stored in `config_train.yaml`.

### Key Points

- **Data Path**  
  - The `data_path` field supports **wildcard patterns**.  
  - You can specify:
    - A single `.h5` file.  
    - Multiple `.h5` files using wildcards (e.g., `../data/*.h5`).  

- **Training Options**  
  - Train from scratch or use pretrained weights for the **VAE**, **FP**, and **DDPM** models.  
  - Training proceeds **sequentially**:
    1. Train the **VAE**.  
    2. Use the trained VAE to train the **FP**.  
    3. Use both the trained VAE and FP to train the **DDPM**.  

- **Pretrained Weights & Epoch Settings**  
  - If pretrained weights are provided for a model **and** `epoch = 0`, training for that model is **skipped**.  
  - If `epoch` is **non-zero**, the model will be trained for the specified number of epochs, regardless of whether pretrained weights are provided.  

- **Training Strategy Summary**  
  - When training begins, a **Training Strategy Summary** is displayed.  
  - This summary shows, for each model (VAE, FP, DDPM):  
    - Whether pretrained weights were loaded.  
    - The number of epochs scheduled for training.  

- **Parameter Details**  
  - A full example config with detailed parameter descriptions is provided below.


---

### 📄 Full Example `config_train.yaml`

```yaml
# ================================
# General settings
# ================================
task: "_"                                # str | default="_" | Task name; auto-generated if blank or "_"
data_path: "../data/experimental/sample_train.h5"  # str | REQUIRED | Path or glob pattern to training dataset (e.g., "../data/.../part_*.h5")
model_dir: "../models/weights/"          # str | default="../models/weights/" | Directory where model weights will be saved after training
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

To use other pretrained weights or datasets, copy the example configurations below into your `config_train.yaml` file. Before doing so, ensure that you have downloaded the corresponding pretrained weights and dataset files as described earlier.

Only update the fields related to the pretrained weights and dataset paths; all other parameters can remain unchanged unless you wish to fine-tune them.

---

**⚠️ Important — Required Parameters for Pretrained Weights**
The following parameters **must match exactly** when using the provided pretrained weights.
If you change any of these values, the pretrained models will not load or function properly.

* `data_path`
* `image_shape`
* `attributes`
* `vae.pretrained_path`
* `vae.latent_dim_channels`
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

## 🧠 Inference

From the repo root:

```bash
cd inference
python inference.py
```
## Inference

This section describes how to run inference using the pretrained weights.

⚠️ **Important:**  
For each pretrained model (**CH 2-Phase**, **CH 3-Phase**, **Experimental**), you must use the **exact** corresponding parameters in the config file as listed in the examples below.  
Changing `attributes`, `image_shape`, or `vae.latent_dim_channels` will result in **incorrect outputs** or **loading errors**.

---

### Output Files

Running inference generates the following outputs:

- **`generated_raw/*.vti`** – Continuous-valued voxel grid.  
  *These `.vti` files can be visualized in ParaView or other similar tools.*

- **`generated_threshold/*.vti`** – Binary mask after thresholding.  
  *Also viewable in ParaView or other similar tools.*

- **CSV inputs** – File containing the input attributes used for generation.

- **CSV outputs** – File containing the generated attributes.

---

### Inference Modes

You can run inference in two ways:

1. **Use Validation Data**  
   - The script reconstructs and generates outputs based on existing dataset samples.

2. **Use Custom Context**  
   - Provide manual context vectors, either:
     - **Single context vector** – Generate multiple outputs under the same conditions.  
     - **Matrix of context vectors** – Generate varied outputs for different conditions.

---

### 📄 Example Configs
#### 📄 Full Example
```yaml
# ================================
# Canonical pretrained compatibility (must match your weights)
# ================================
data_path: "../../../total_dataset_for_hugging_face/microgen3D/tmp_data/CH_three_phase/val/part_*.h5"   # str | REQUIRED for inference.mode=dataset | Path or glob to data (e.g., "../.../part_*.h5"); set null for other modes
batch_size: 2                                      # int | default=20 | Batch size used during generation and dataset loading
image_shape: [1, 128, 128, 64]                        # list[int] | default=[1,64,64,64] | Input volume shape [C, D, H, W]
attributes:                                         # list[str] | REQUIRED | Full FP output order; order must match the weights
  - vol_frac_D
  - vol_frac_M
  - tortuosity_A
  - tortuosity_D
  - phi
  - chi
  - log_time

# ================================
# VAE settings (used only for loading + decoding)
# ================================
vae:
  latent_dim_channels: 4                            # int | default=1 | Latent channel count used during training
  kld_loss_weight: 0.000001                         # float | default=1e-6 | Not used at inference; kept for compatibility
  max_epochs: 0                                     # int | default=0 | Leave at 0 for inference (no training)
  pretrained_path: "../models/weights/CH_3phase/vae.pt"  # str | default="" | Path to VAE weights; must exist for inference
  first_layer_downsample: false                      # bool | default=False | Must match training config; affects encoder stride
  max_channels: 512                                 # int | default=512 | Architecture width; must match training

# ================================
# FP (feature predictor) settings
# ================================
fp:
  max_epochs: 0                                     # int | default=0 | Leave at 0 for inference (no training)
  pretrained_path: "../models/weights/CH_3phase/fp.pt"   # str | default="" | Path to FP weights; must exist for inference

# ================================
# DDPM (generator) settings
# ================================
ddpm:
  timesteps: 1000                                   # int | default=1000 | Diffusion steps; must match training for best results
  n_feat: 512                                       # int | default=512 | UNet base width; must match training
  max_epochs: 0                                     # int | default=0 | Leave at 0 for inference (no training)
  pretrained_path: "../models/weights/CH_3phase/ddpm.pt" # str | default="" | Path to DDPM weights; must exist for inference
  context_attributes:                               # list[str] | default=<attributes> | Subset (in order) used as DDPM conditioning
    - vol_frac_D
    - vol_frac_M
    - tortuosity_A
    - tortuosity_D

# ================================
# Inference controls (choose how to provide context)
# ================================
inference:
  mode: "dataset"                                  # str | default="constant" | One of: "constant" | "random" | "dataset"
  total_samples: 4                                # int | default=100 | Total number of generated samples

  # --- mode="constant": broadcast a single context row to the whole batch ---
  constant_context:                                 # list[float] or list[int] | default=[] | Either full attributes or just context_attributes order
    - 0.5
    - 0.2
    - 0.2

  # --- mode="random": sample contexts uniformly within per-attribute ranges ---
  random:
    ranges:                                         # dict[str -> [float lo, float hi]] | REQUIRED for mode="random"
      ABS_f_D: [0.0, 0.96]
      CT_f_D_tort1: [0.05, 0.95]
      CT_f_A_tort1: [0.05, 0.95]

  # --- mode="dataset": derive context from FP( VAE(latent(x)) ) using a dataloader ---
  dataset_loader: "val"                             # str | default="val" | One of: "train" | "val"; requires data_path to be valid

# ================================
# Output controls
# ================================
output:
  output_dir: "./output"                            # str | default="./output" | Base directory for all outputs
  write_vti: true                                   # bool | default=true | Write .vti volumes (generated & optionally original/recon)
  write_csv: true                                   # bool | default=true | Write CSVs for input contexts and predicted features
  threshold: 0.5                                    # float | default=0.5 | Threshold for *_threshold volumes
  save_every_batch: true                            # bool | default=true | Flush CSVs after each batch (safer for long runs)
  csv_inputs: "inputs_context.csv"                  # str | default="inputs_context.csv" | CSV filename for the used contexts (context_attributes columns)
  csv_outputs: "outputs_predicted.csv"              # str | default="outputs_predicted.csv" | CSV filename for FP predictions (full attributes columns)

```

#### CH 2-Phase

```yaml
data_path: "../data/sample_CH_two_phase/train/part_*.h5"  # wildcard to use all parts
image_shape: [1, 128, 128, 64]
attributes:
  - norm_STAT_e
  - ABS_f_D
  - ABS_f_D
  - DISS_wf10_D
  - CT_f_e_conn
  - CT_f_D_tort1
  - CT_f_A_tort1

vae.pretrained_path: "../models/weights/CH_2phase/vae.pt"
vae.latent_dim_channels: 4
fp.pretrained_path: "../models/weights/CH_2phase/fp.pt"
ddpm.pretrained_path: "../models/weights/CH_2phase/ddpm.pt"
ddpm.context_attributes:
  - ABS_f_D
  - CT_f_D_tort1
  - CT_f_A_tort1
```

---

#### CH 3-Phase

```yaml
data_path: "../data/sample_CH_three_phase/train/part_*.h5"  # wildcard to use all parts
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
fp.pretrained_path: "../models/weights/CH_3phase/fp.pt"
ddpm.pretrained_path: "../models/weights/CH_3phase/ddpm.pt"
ddpm.context_attributes:
  - vol_frac_D
  - vol_frac_M
  - tortuosity_A
  - tortuosity_D
```

---

#### Experimental

```yaml
data_path: "../data/experimental/train/train.h5"  # Path to the experimental dataset
image_shape: [1, 64, 64, 64]
attributes:
  - ABS_f_D
  - CT_f_D_tort1
  - CT_f_A_tort1

vae.pretrained_path: "../models/weights/experimental/vae.pt"
vae.latent_dim_channels: 1
fp.pretrained_path: "../models/weights/experimental/fp.pt"
ddpm.pretrained_path: "../models/weights/experimental/ddpm.pt"
ddpm.context_attributes:
  - ABS_f_D
  - CT_f_D_tort1
  - CT_f_A_tort1
```




## 📄 License

[MIT License](LICENSE)
