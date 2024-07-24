from itertools import combinations
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

from torch.utils.data import Dataset, IterableDataset, random_split
from torch.utils.data.dataset import Subset
from torchvision import transforms
from torchvision.datasets.mnist import MNIST


mnist_raw_transform = transforms.Compose([
    # transforms.Resize((32, 32)),  # Resize the images to 32x32 pixels
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=(0.1307,), std=(0.3081,))  # Normalize the images with mean and standard deviation
])

class MNISTDataset(Dataset):
    def __init__(self, dset: torch.tensor, labels: torch.tensor):
        self.dset, self.labels = dset, labels

    def __len__(self):
        return len(self.dset)

    def __getitem__(self, idx: int):
        return self.dset[idx].float().unsqueeze(0), self.labels[idx]

    def transform_raw(self):
        return transforms.Compose([
            # transforms.Resize((32, 32)),  # Resize the images to 32x32 pixels
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))  # Normalize the images with mean and standard deviation
        ])


class MNISTDatasets:
    """Manages Torch Datasets for training, validation, testing for various tasks."""

    def __init__(self, train: MNIST, test: MNIST):
        self.train = train.data
        self.train_labels = train.train_labels
        self.test = test.data
        self.test_labels = test.test_labels

    def get_standard_torch_dset(self, split: str, num_train=None) -> Dataset:
        if split not in ["train", "valid", "test"]:
            raise ValueError("Please enter a valid split!")
        elif split == "test":
            dset = MNISTDataset(self.test, self.test_labels)
        else:
            train, valid = self.get_train_validation_split(num_train)
            if split == "train":
                dset = MNISTDataset(train.dataset, train.labels)
            else:
                dset = MNISTDataset(valid.dataset, valid.labels)

        return dset

    def get_train_validation_split(self, num_train: int) -> list[Subset, Subset]:
        split_num = (num_train, len(self.train) - num_train)
        # print("type of dset:", type(self.train))
        # print("split num:", split_num)
        splits = random_split(self.train, split_num)     # returns list containing two Subsets
        # print("train split:", len(splits[0]))
        # print("train should be 50000... ", splits[0].dataset.shape)

        # select only those observations to keep
        for dset in splits:
            idx = dset.indices
            dset.dataset = dset.dataset[idx]
            labels = self.train_labels[idx]
            dset.labels = labels

        return splits

    def construct_block_dataset(self, dset: torch.tensor, labels: torch.tensor,
                                n_samples=None, dist_shift=False) -> MNISTDataset:
        """Constructs 2x2 MNIST block and corresponding label for PVR task."""
        num_regions = 4
        dset_size = dset.size(0)
        n_samples = dset_size if n_samples is None else n_samples
        rand_indices = torch.randint(0, dset_size, size=(n_samples, num_regions))

        # create grid
        Data = namedtuple("Data", ["X", "Y"])
        TL, TR, BL, BR = [Data(dset[rand_indices[:, region]],
                               labels[rand_indices[:, region]])
                          for region in range(num_regions)]

        # recall shape: (len(dset), 28, 28)
        top = torch.cat([TL.X, TR.X], dim=2)    # (len(dset), 28, 56)
        bot = torch.cat([BL.X, BR.X], dim=2)
        grid = torch.cat([top, bot], dim=1)

        # compute labels
        if dist_shift:
            # TODO
            raise NotImplementedError
        else:
            pointers = TL.Y
            labels = torch.empty((n_samples,), dtype=torch.long)

            TR_mask = torch.isin(pointers, torch.tensor([0,1,2,3]))
            BL_mask = torch.isin(pointers, torch.tensor([4,5,6]))
            BR_mask = torch.isin(pointers, torch.tensor([7,8,9]))

            labels[TR_mask] = TR.Y[TR_mask]
            labels[BL_mask] = BL.Y[BL_mask]
            labels[BR_mask] = BR.Y[BR_mask]

        return MNISTDataset(grid, labels)


class MNISTPaired(IterableDataset):
    def __init__(self, split_dset: torch.tensor, split_labels: torch.tensor,
                 train_idx: torch.tensor = None, max_iter=1e6):
        self.split_dset = split_dset
        self.split_labels = split_labels
        if train_idx is not None:
            self.pairs = combinations(train_idx.tolist(), 2)
        else:
            self.pairs = combinations(range(len(split_dset)), 2)

    def __iter__(self):
        return iter(self.pairs)


class MNISTDoublePaired(IterableDataset):
    def __init__(self, split_dset1, split_dset2,
                 split_labels1, split_labels2,
                 train_idx1, train_idx2):
        self.pairs1 = MNISTPaired(split_dset1, split_labels1, train_idx1)
        self.pairs2 = MNISTPaired(split_dset2, split_labels2, train_idx2)

    def __iter__(self):
        return iter(zip(self.pairs1, self.pairs2))



if __name__ == "__main__":
    DATA_DIR = Path("data")
    train = MNIST(DATA_DIR, transform=transforms.ToTensor(), train=True, download=True)
    test = MNIST(DATA_DIR, transform=transforms.ToTensor(), train=False, download=True)
    datasets = MNISTDatasets(train, test)

    # torch.utils.data.dataset.Subset with .dataset and .labels properties
    train_dset, valid_dset = datasets.get_train_validation_split(50000)
    block_dset, labels = datasets.construct_block_dataset(train_dset.dataset,
                                                          train_dset.labels)
