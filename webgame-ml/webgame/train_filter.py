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
    def __init__(
        self,
        channels: int,
        size: int,
        use_pos: bool = False,
        objs_shape: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        num_channels = channels
        if use_pos:
            num_channels += 2
        self.use_objs = objs_shape is not None

        # Grid + scalar feature processing
        grid_features_dim = 32
        self.grid_net = nn.Sequential(
            nn.Conv2d(num_channels, 32, 3, padding="same", dtype=torch.double),
            nn.BatchNorm2d(32, dtype=torch.double),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding="same", dtype=torch.double),
            nn.BatchNorm2d(32, dtype=torch.double),
            nn.SiLU(),
            nn.Conv2d(32, grid_features_dim, 3, padding="same", dtype=torch.double),
            nn.BatchNorm2d(32, dtype=torch.double),
            nn.SiLU(),
        )

        # Positional encoding
        x_channel = torch.tensor([list(range(size))] * size, dtype=torch.double) / size
        y_channel = x_channel.T
        self.pos = torch.stack(
            [x_channel, y_channel]
        )  # Shape: (2, grid_size, grid_size)
        self.pos.requires_grad = (
            False  # TODO: Double check that the positional encoding doesn't change
        )
        self.use_pos = use_pos

        # Fusion of object features into grid + scalar features
        if objs_shape:
            _, obj_dim = objs_shape
            proj_dim = 64
            n_heads = 4
            self.proj = nn.Conv1d(obj_dim, proj_dim, 1)
            self.attn1 = nn.MultiheadAttention(
                proj_dim, n_heads, batch_first=True, dtype=torch.double
            )
            self.bn1 = nn.BatchNorm1d(grid_features_dim)
            self.attn2 = nn.MultiheadAttention(
                proj_dim, n_heads, batch_first=True, dtype=torch.double
            )
            self.bn2 = nn.BatchNorm1d(grid_features_dim)
            self.attn3 = nn.MultiheadAttention(
                proj_dim, n_heads, batch_first=True, dtype=torch.double
            )
            self.bn3 = nn.BatchNorm1d(grid_features_dim)

        # Convert features into liklihood map
        self.out_net = nn.Sequential(
            nn.Conv2d(grid_features_dim, 32, 3, padding="same", dtype=torch.double),
            nn.BatchNorm2d(32, dtype=torch.double),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding="same", dtype=torch.double),
            nn.BatchNorm2d(32, dtype=torch.double),
            nn.SiLU(),
            nn.Conv2d(32, 1, 3, padding="same", dtype=torch.double),
            nn.Sigmoid(),
        )

    def forward(
        self,
        grid: Tensor,  # Shape: (batch_size, channels, size, size)
        objs: Tensor,  # Shape: (batch_size, max_obj_size, obj_dim)
        objs_attn_mask: Tensor,  # Shape: (batch_size, max_obj_size)
    ) -> Tensor:
        # Concat pos encoding to grid
        device = grid.device
        if self.use_pos:
            grid = torch.concat(
                [
                    grid,
                    self.pos.unsqueeze(0)
                    .tile([grid.shape[0], 1, 1, 1])
                    .to(device=device),
                ],
                dim=1,
            )
        grid_features = self.grid_net(
            grid
        )  # Shape: (batch_size, grid_features_dim, grid_size, grid_size)

        if self.use_objs:
            objs = self.proj(objs.permute(0, 2, 1)).permute(
                0, 2, 1
            )  # Shape: (batch_size, max_obj_size, proj_dim)
            grid_features = grid_features.permute(
                0, 2, 3, 1
            )  # Shape: (batch_size, grid_size, grid_size, grid_features_dim)
            orig_shape = grid_features.shape
            grid_features = grid_features.flatten(
                1, 2
            )  # Shape: (batch_size, grid_size * grid_size, grid_features_dim)
            attns = [self.attn1, self.attn2, self.attn3]
            bns = [self.bn1, self.bn2, self.bn3]
            for attn, bn in zip(attns, bns):
                grid_features = attn(
                    query=grid_features,
                    key=objs,
                    value=objs,
                    key_padding_mask=objs_attn_mask,
                )
                grid_features = bn(grid_features.permute(0, 2, 1)).permute(0, 2, 1)
                grid_features = nn.functional.silu(grid_features)
            grid_features = grid_features.reshape(orig_shape).permute(
                0, 3, 1, 2
            )  # Shape: (batch_size, grid_features_dim, grid_size, grid_size)

        return self.out_net(grid_features).squeeze(1) # Shape: (batch_size, grid_size, grid_size)


def predict(belief: Tensor) -> Tensor:
    kernel = torch.tensor(
        [[[[0, 1, 0], [1, 1, 1], [0, 1, 0]]]], dtype=belief.dtype, device=belief.device
    )
    kernel = kernel / kernel.sum()
    belief = torch.nn.functional.conv2d(belief.unsqueeze(1), kernel, padding="same")
    return belief / belief.sum()


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--traj-dir", type=str)
    parser.add_argument("--out-dir", type=str, default="./runs")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--use-pos", default=False, action="store_true")
    parser.add_argument("--use-objs", default=False, action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)

    with open(Path(args.traj_dir) / "traj_data_all.pkl", "rb") as f:
        traj_data_all: TrajDataAll = pkl.load(f)

    out_dir = Path(args.out_dir)
    out_id = "".join(
        [random.choice(string.ascii_letters + string.digits) for _ in range(8)]
    )
    os.mkdir(out_dir / out_id)
    chkpt_path = out_dir / out_id / "checkpoints"
    os.mkdir(chkpt_path)

    ds_x = torch.tensor(
        traj_data_all.seqs, device=device, dtype=torch.double
    )  # Shape: (num_seqs, seq_len, channels, grid_size, grid_size)
    num_seqs, seq_len, channels, grid_size = ds_x.shape[:-1]
    ds_y = torch.tensor(
        [
            [tile[1] * grid_size + tile[0] for tile in tiles]
            for tiles in traj_data_all.tiles
        ],
        dtype=torch.long,
        device=device,
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
    model = MeasureModel(channels, grid_size, args.use_pos)
    model.to(device=device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    wandb_config = {
        "experiment": "bayes",
        "train_size": num_seqs_train,
        "valid_size": num_seqs_valid,
        "grid_size": grid_size,
    }
    wandb_config.update(args.__dict__)
    wandb.init(project="pursuer", config=wandb_config)

    batch_size = args.batch_size
    batches_per_epoch = num_seqs_train // batch_size
    for epoch in tqdm(range(args.epochs)):
        model.train()
        seq_idxs = torch.randperm(num_seqs_train, device=device)
        avg_loss = 0.0
        for batch_idx in range(batches_per_epoch):
            batch_x = train_x[
                seq_idxs[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            ]
            batch_y = train_y[
                seq_idxs[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            ]
            priors = (
                torch.ones(
                    [batch_size, grid_size**2], dtype=torch.double, device=device
                )
                / grid_size**2
            )
            loss = torch.zeros([1], device=device)
            for step in range(seq_len):
                lkhd = torch.reshape(
                    model(batch_x[:, step, :, :, :]), [batch_size, grid_size**2]
                )
                new_priors = predict(
                    priors.reshape([batch_size, grid_size, grid_size])
                ).reshape([batch_size, grid_size**2])
                new_priors = new_priors * lkhd
                new_priors = new_priors / new_priors.sum(1, keepdim=True)
                priors = new_priors
                loss += ce(priors, batch_y[:, step])
            opt.zero_grad()
            loss.backward()
            opt.step()

            avg_loss += loss.item()

        model.eval()
        with torch.no_grad():
            avg_valid_loss = 0.0
            priors = (
                torch.ones(
                    [num_seqs_valid, grid_size**2], dtype=torch.double, device=device
                )
                / grid_size**2
            )
            for step in range(seq_len):
                lkhd = torch.reshape(
                    model(valid_x[:, step, :, :, :]), [num_seqs_valid, grid_size**2]
                )
                new_priors = predict(
                    priors.reshape([num_seqs_valid, grid_size, grid_size])
                ).reshape([num_seqs_valid, grid_size**2])
                new_priors = new_priors * lkhd
                new_priors = new_priors / new_priors.sum(1, keepdim=True)
                priors = new_priors
                avg_valid_loss += ce(priors, valid_y[:, step]).item()

        avg_loss = avg_loss / batches_per_epoch

        wandb.log({"train_loss": avg_loss, "valid_loss": avg_valid_loss})

        if epoch % args.save_every == 0:
            save_model(model, str(chkpt_path / f"model-{epoch}.safetensors"))


if __name__ == "__main__":
    main()
