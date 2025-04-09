# MicroGen3D

 MicroGen3D is a conditional latent diffusion model framework for generating high-resolution 3D multiphase microstructures with user-defined attributes such as volume fraction and tortuosity. Designed to accelerate materials discovery, it can synthesizes microstructures within a few seconds and predicts associated manufacturing parameters.

## 📦 Setup Instructions

### 1. Clone the Repository
```bash
git clone https://your-repo-url.git
cd your-repo-name
```

### 2. Set Up Python Virtual Environment
Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Required Packages
```bash
pip install -r requirements.txt
```

## 📁 Download Sample Data and Pretrained Weights

Download the representative data and pretrained model weights from:  
**[https://mydatalinkbluh.com](https://mydatalinkbluh.com)**

Place the following files into the `data/` directory:
- `sample_train.h5`

Place pretrained model checkpoints into `models/weights/`.

## ⚙️ Configuration

### Training Config (`config.yaml`)
- **task**: Auto-generated if left null  
- **data_path**: Path to training dataset (`../data/sample_train.h5`)  
- **model_dir**: Directory to save model weights  
- **batch_size**: Batch size for training  
- **image_shape**: Shape of the 3D images `[C, D, H, W]`  

#### VAE Settings:
- `latent_dim_channels`: Latent space size  
- `kld_loss_weight`: Weight of KL divergence loss  
- `max_epochs`: Training epochs  
- `pretrained`: Whether to use pretrained VAE  
- `pretrained_path`: Path to pretrained VAE model  

#### FP Settings:
- `dropout`: Dropout rate  
- `max_epochs`: Training epochs  
- `pretrained`: Whether to use pretrained FP  
- `pretrained_path`: Path to pretrained FP model  

#### DDPM Settings:
- `timesteps`: Number of diffusion timesteps  
- `n_feat`: Number of base features  
- `learning_rate`: Learning rate  
- `max_epochs`: Training epochs  

### Inference Parameters (`params.yaml`)
- **data_path**: Path to inference/test dataset (`../data/sample_test.h5`)  

#### Training (for model init only):
- `batch_size`, `num_batches`, `num_timesteps`, `learning_rate`, `max_epochs`  

#### Model:
- `n_feat`: Number of base features  
- `image_shape`: Expected image input shape  

#### Attributes:
- List of features/targets to predict:
  - `ABS_f_D`
  - `CT_f_D_tort1`
  - `CT_f_A_tort1`

#### Paths:
- `ddpm_path`: Path to trained DDPM model  
- `vae_path`: Path to trained VAE model  
- `fc_path`: Path to trained FP model  
- `output_dir`: Where to store inference results  

## 🏋️ Training

Navigate to the training folder and run:
```bash
cd training
python training.py
```

## 🧠 Inference

After training, switch to the inference folder and run:
```bash
cd ../inference
python inference.py
```

Make sure the paths in `params.yaml` are correctly set and pretrained models are placed in `models/weights/`.

## 📌 Notes

- Sample data and pretrained models must be downloaded from [https://mydatalinkbluh.com](https://mydatalinkbluh.com)  
- Model outputs will be saved in the folder specified by `output_dir` in `params.yaml`  
- Image shape and features must be consistent across config files and dataset format  
```


