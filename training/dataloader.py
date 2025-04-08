import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
import h5py

class H5Dataset(Dataset):
    def __init__(self, file_path, transform=None):
        super(H5Dataset, self).__init__()
        self.file_path = file_path
        with h5py.File(file_path, 'r') as h5_file:
            self.keys = list(h5_file.keys())
        self.transform = transform

    def __getitem__(self, index):
        with h5py.File(self.file_path, 'r') as h5_file:
            # Load image data
            image = torch.from_numpy(h5_file[self.keys[index]][:].reshape(1, 64,64,64)).float()


        return image 

    def __len__(self):
        return len(self.keys)

class ImageDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, data_path, transform=None):
        super().__init__()
        self.batch_size = batch_size
        self.data_path = data_path
        self.transform = transform

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            full_dataset = H5Dataset(self.data_path, transform=self.transform)
            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)



