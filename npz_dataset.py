"""
npz_dataset.py

Dataset class (for PyTorch DataLoader) for data saved in *.npz or *.npy format.
If using a *.npz file, it must contain an array 'x' that stores all the data and
can contain an optional array 'params' of known parameters for comparison.
"""

import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader

class PDEDataset(Dataset):
    """PDE dataset with inputs x and targets also x."""

    def __init__(self, data_file=None, transform=None, data_size=None):
        """
        Args:
            data_file (numpy save): file with all data
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        data = np.load(data_file)

        if type(data) is np.ndarray:
            self.data_x = data
            self.params = None
        elif 'x' in data.files:
            self.data_x = data['x']
            self.params = data['params'] if 'params' in data.files else None
        else:
            raise ValueError("Dataset import failed. NPZ files must include 'x' array containing data.")

        self.transform = transform


    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):

        x = torch.from_numpy(self.data_x[idx])

        if self.params is None:
            sample = [x, x, torch.tensor(float('nan'))]
        else:
            sample = [x, x, torch.from_numpy(self.params[idx])]

        if self.transform:
            sample = self.transform(sample)

        return sample
