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

        self.data_x = data['x']
        self.params = data['params']

        self.transform = transform


    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):

        x = torch.from_numpy(self.data_x[idx])
        sample = [x, x, torch.from_numpy(self.params[idx])]

        if self.transform:
            sample = self.transform(sample)

        return sample
