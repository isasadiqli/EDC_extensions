import os
import numpy as np
import torch
from torch.utils.data import Dataset


class MRNetDataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        self.scans = self.load_scans()

    def load_scans(self):
        scans = []
        label_file = 'train-labels.csv' if self.train else 'valid-labels.csv'
        with open(os.path.join(self.root_dir, label_file), 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                file_name, label = line.strip().split(',')
                file_path = os.path.join(self.root_dir, 'train' if self.train else 'valid', file_name + '.npy')
                scans.append((file_path, int(label)))
        return scans

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, idx):
        file_path, label = self.scans[idx]
        volume = np.load(file_path)  # Loading the 3D volume from .npy file
        volume = torch.tensor(volume, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        if self.transform:
            volume = self.transform(volume)

        return volume, label


class MRNetSliceDataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        self.scans = self.load_scans()

    def load_scans(self):
        scans = []
        label_file = 'train-labels.csv' if self.train else 'valid-labels.csv'
        with open(os.path.join(self.root_dir, label_file), 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                file_name, label = line.strip().split(',')
                file_path = os.path.join(self.root_dir, 'train' if self.train else 'valid', file_name + '.npy')
                scans.append((file_path, int(label)))
        return scans

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, idx):
        file_path, label = self.scans[idx]
        volume = np.load(file_path)  # Loading the 3D volume from .npy file
        slices = [torch.tensor(volume[:, :, i], dtype=torch.float32).unsqueeze(0) for i in range(volume.shape[2])]

        if self.transform:
            slices = [self.transform(slice) for slice in slices]

        return slices, label




