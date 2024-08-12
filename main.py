import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim

from loader import RobustImageFolder
from models import GhostImageNet
from trainer import Trainer

from constants import TRANSFORM


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("train_data_path", type=Path)
    parser.add_argument("--data_split", default=0.8, type=float)
    parser.add_argument("--data_split_seed", default=42, type=int)
    parser.add_argument("--torch_seed", default=42, type=int)
    parser.add_argument("--learning_rate", default=0.001, type=float)
    parser.add_argument("--parallel", default=True, type=bool)

    return parser.parse_args()


def get_cuda_devices():
    assert torch.cuda.is_available()
    return [idx for idx in range(torch.cuda.device_count())]


def configure_seed_globally(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):
    train_data_path: Path = args.train_data_path
    data_split_rate: float = args.data_split
    data_split_seed: int = args.data_split_seed
    torch_seed: int = args.torch_seed
    lr: float = args.learning_rate
    parallel: bool = args.parallel

    if not torch.cuda.is_available():
        raise Exception("Cuda is not able to use. Please run on the cuda device")

    configure_seed_globally(torch_seed)

    full_dataset = RobustImageFolder(train_data_path, seed=data_split_seed, transform=TRANSFORM)
    train_loader, val_loader = full_dataset.generate_data_loader(data_split_rate)

    model = GhostImageNet(1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if parallel:
        model = nn.DataParallel(model, device_ids=get_cuda_devices())
        model = model.to(torch.device("cuda"))

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
    )


if __name__ == "__main__":
    main(parse_args())
