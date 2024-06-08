"""
Script for training our filter.
"""

from argparse import ArgumentParser
import os
from pathlib import Path
import random
import string
from typing import *
import pickle as pkl

import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from torch import Tensor, nn
import torch
from safetensors.torch import save_model

from webgame.gen_trajectories import TrajDataAll
import wandb


class MeasureModel(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding="same", dtype=torch.double),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding="same", dtype=torch.double),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding="same", dtype=torch.double),
            nn.SiLU(),
            nn.Conv2d(32, 1, 3, padding="same", dtype=torch.double),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.net(x).squeeze(1)  # Shape: (batch_size, grid_size, grid_size)
        return x

def predict(belief: Tensor) -> Tensor:
    kernel = torch.tensor([[[[0, 1, 0], [1, 1, 1], [0, 1, 0]]]], dtype=belief.dtype, device=belief.device)
    kernel = kernel / kernel.sum()
    belief = torch.nn.functional.conv2d(belief.unsqueeze(1), kernel, padding="same")
    return belief / belief.sum()

def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--traj-dir", type=str, default="./runs/bCoBe4mT")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    wandb_config = {
        "experiment": "bayes",
    }
    wandb_config.update(args.__dict__)
    wandb.init(
        project="pursuer",
        config=wandb_config
    )

    chkpt_path = Path(args.traj_dir) / "checkpoints"
    os.mkdir(chkpt_path)

    with open(Path(args.traj_dir) / "traj_data_all.pkl", "rb") as f:
        traj_data_all: TrajDataAll = pkl.load(f)

    ds_x = torch.tensor(
        traj_data_all.seqs,
        device=device,
        dtype=torch.double
    )  # Shape: (num_seqs, seq_len, channels, grid_size, grid_size)
    num_seqs, seq_len, channels, grid_size = ds_x.shape[:-1]
    ds_y = torch.tensor(
        [
            [tile[1] * grid_size + tile[0] for tile in tiles]
            for tiles in traj_data_all.tiles
        ],
        dtype=torch.long,
        device=device
    )  # Shape: (num_seqs, seq_len)
    valid_pct = 0.2
    num_seqs_valid = int(num_seqs * valid_pct)
    num_seqs_train = num_seqs - num_seqs_valid
    train_x = ds_x[:num_seqs_train]
    train_y = ds_y[:num_seqs_train]
    valid_x = ds_x[num_seqs_train:]
    valid_y = ds_y[num_seqs_train:]
    del traj_data_all, ds_x, ds_y
    ce = nn.CrossEntropyLoss()
    model = MeasureModel(channels)
    model.to(device=device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    batch_size = args.batch_size
    batches_per_epoch = args.epochs // batch_size
    for epoch in tqdm(range(args.epochs)):
        seq_idxs = torch.randperm(num_seqs_train, device=device)
        avg_loss = 0.0
        for batch_idx in range(batches_per_epoch):
            batch_x = train_x[
                seq_idxs[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            ]
            batch_y = train_y[
                seq_idxs[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            ]
            priors = torch.ones([batch_size, grid_size**2], dtype=torch.double, device=device) / grid_size**2
            loss = torch.zeros([1], device=device)
            for step in range(seq_len):
                lkhd = torch.reshape(model(batch_x[:, step, :, :, :]), [batch_size, grid_size**2])
                new_priors = predict(priors.reshape([batch_size, grid_size, grid_size])).reshape([batch_size, grid_size**2])
                new_priors = (new_priors * lkhd)
                new_priors = new_priors / new_priors.sum(1, keepdim=True)
                priors = new_priors
                loss += ce(lkhd, batch_y[:, step])
            opt.zero_grad()
            loss.backward()
            opt.step()

            avg_loss += loss.item()
        
        with torch.no_grad():
            avg_valid_loss = 0.0
            priors = torch.ones([num_seqs_valid, grid_size**2], dtype=torch.double, device=device) / grid_size**2
            for step in range(seq_len):
                lkhd = torch.reshape(model(valid_x[:, step, :, :, :]), [num_seqs_valid, grid_size**2])
                new_priors = predict(priors.reshape([num_seqs_valid, grid_size, grid_size])).reshape([num_seqs_valid, grid_size**2])
                new_priors = (new_priors * lkhd)
                new_priors = new_priors / new_priors.sum(1, keepdim=True)
                priors = new_priors
                avg_valid_loss += ce(lkhd, valid_y[:, step]).item()

        avg_loss = avg_loss / batches_per_epoch

        wandb.log({
            "train_loss": avg_loss,
            "valid_loss": avg_valid_loss
        })

        if epoch % args.save_every == 0:
            save_model(model, str(chkpt_path / f"model-{epoch}.safetensors"))


if __name__ == "__main__":
    main()
