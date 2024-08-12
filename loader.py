import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader


class RobustImageFolder(ImageFolder):
    def __init__(self, root, split_seed: int, transform=None, batch_size=128):
        super().__init__(root, transform=transform, loader=self.robust_loader)
        self.transform = transform
        self.split_seed = split_seed
        self.batch_size = batch_size

    def robust_loader(self, path):
        try:
            return default_loader(path)
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            return None

    def __getitem__(self, index):
        while True:
            # print(index)
            path, target = self.samples[index]
            sample = self.loader(path)
            if sample is None:
                print(f"Skipping corrupted image: {path}")
                index = (index + 1) % len(self.samples)
            else:
                break
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

    def split_data_loader(self, rate: float = 0.8):
        total_image_count = len(self)
        all_indicies = np.arange(total_image_count)
        np.random.seed(self.split_seed)
        random_indicies = np.random.shuffle(all_indicies)

        split = int(rate * total_image_count)

        train_indices = random_indicies[split:]
        val_indices = random_indicies[:split]

        train_dataset = Subset(self, train_indices)
        val_dataset = Subset(self, val_indices)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
        )
        return train_loader, val_loader
