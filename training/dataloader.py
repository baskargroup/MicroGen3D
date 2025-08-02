import glob
import os
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class H5Dataset(Dataset):
    def __init__(self, file_pattern, attributes, image_shape, transform=None):
        super().__init__()
        self.file_paths = sorted(glob.glob(file_pattern))  # List of .h5 files
        self.attributes = attributes
        self.image_shape = image_shape
        self.transform = transform

        # Collect all (file_index, key) pairs
        self.index_map = []
        for file_idx, file_path in enumerate(self.file_paths):
            with h5py.File(file_path, 'r') as f:
                keys = list(f.keys())
                self.index_map.extend([(file_idx, key) for key in keys])

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, index):
        file_idx, key = self.index_map[index]
        file_path = self.file_paths[file_idx]

        with h5py.File(file_path, 'r') as f:
            image = torch.from_numpy(f[key][:].reshape(*self.image_shape)).float()
            attrs = f[key].attrs
            label = torch.tensor([attrs[attr] for attr in self.attributes], dtype=torch.float)

        if self.transform:
            image = self.transform(image)

        return image, label

class ImageDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, data_path, attributes, image_shape, transform=None):
        super().__init__()
        self.batch_size = batch_size
        self.data_path = data_path  # This can be a glob pattern now
        self.attributes = attributes
        self.image_shape = image_shape
        self.transform = transform

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            full_dataset = H5Dataset(
                file_pattern=self.data_path,
                attributes=self.attributes,
                image_shape=self.image_shape,
                transform=self.transform
            )

            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size]
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
