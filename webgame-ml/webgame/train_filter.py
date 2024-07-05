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

from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from torch import Tensor, nn
import torch
from safetensors.torch import save_model

from webgame.gen_trajectories import TrajDataAll
import wandb

from webgame.models import MeasureModel


def predict(belief: Tensor) -> Tensor:
    kernel = torch.tensor(
        [[[[0, 1, 0], [1, 1, 1], [0, 1, 0]]]], dtype=belief.dtype, device=belief.device
    )
    kernel = kernel / kernel.sum()
    belief = torch.nn.functional.conv2d(belief.unsqueeze(1), kernel, padding="same")
    return belief


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--traj-dir", type=str)
    parser.add_argument("--out-dir", type=str, default="./runs")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--use-pos", default=False, action="store_true")
    parser.add_argument("--use-objs", default=False, action="store_true")
    parser.add_argument("--lkhd-min", type=float, default=0.001)
    parser.add_argument("--only-opt-last", default=False, action="store_true")

    args = parser.parse_args()
    device = torch.device(args.device)
    lkhd_min = args.lkhd_min

    with open(Path(args.traj_dir) / "traj_data_all.pkl", "rb") as f:
        traj_data_all: TrajDataAll = pkl.load(f)

    ds_x_grid = torch.tensor(
        [[x[0] for x in xs] for xs in traj_data_all.seqs],
        device=device,
        dtype=torch.float,
    )  # Shape: (num_seqs, seq_len, channels, grid_size, grid_size)
    ds_x_objs = torch.tensor(
        [[x[1] for x in xs] for xs in traj_data_all.seqs],
        device=device,
        dtype=torch.float,
    )  # Shape: (num_seqs, seq_len, max_objs, obj_dim)
    ds_x_mask = torch.tensor(
        [[x[2] for x in xs] for xs in traj_data_all.seqs],
        device=device,
        dtype=torch.bool,
    )  # Shape: (num_seqs, seq_len, max_objs)
    num_seqs, seq_len, channels, grid_size = ds_x_grid.shape[:-1]
    max_objs, obj_dim = ds_x_objs.shape[2:]
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
    train_x_grid = ds_x_grid[:num_seqs_train]
    train_x_objs = None
    train_x_mask = None
    if args.use_objs:
        train_x_objs = ds_x_objs[:num_seqs_train]
        train_x_mask = ds_x_mask[:num_seqs_train]
    train_y = ds_y[:num_seqs_train]
    valid_x_grid = ds_x_grid[num_seqs_train:]
    if args.use_objs:
        valid_x_objs = ds_x_objs[num_seqs_train:]
        valid_x_mask = ds_x_mask[num_seqs_train:]
    valid_y = ds_y[num_seqs_train:]
    del traj_data_all, ds_x_grid, ds_x_objs, ds_x_mask, ds_y
    nll = nn.NLLLoss()
    model = MeasureModel(
        channels,
        grid_size,
        args.use_pos,
        (max_objs, obj_dim) if args.use_objs else None,
    )
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

    assert wandb.run is not None
    while wandb.run.name == "":
        pass
    out_id = wandb.run.name

    out_dir = Path(args.out_dir)
    os.mkdir(out_dir / out_id)
    chkpt_path = out_dir / out_id / "checkpoints"
    os.mkdir(chkpt_path)

    batch_size = args.batch_size
    batches_per_epoch = num_seqs_train // batch_size
    for epoch in tqdm(range(args.epochs)):
        model.train()
        seq_idxs = torch.randperm(num_seqs_train, device=device)
        avg_loss = 0.0
        avg_loss_all = 0.0
        for batch_idx in range(batches_per_epoch):
            batch_x_grid = train_x_grid[
                seq_idxs[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            ]
            batch_x_objs = None
            batch_x_mask = None
            if args.use_objs:
                assert train_x_objs is not None
                assert train_x_mask is not None
                batch_x_objs = train_x_objs[
                    seq_idxs[batch_idx * batch_size : (batch_idx + 1) * batch_size]
                ]
                batch_x_mask = train_x_mask[
                    seq_idxs[batch_idx * batch_size : (batch_idx + 1) * batch_size]
                ]
            batch_y = train_y[
                seq_idxs[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            ]
            priors = (
                torch.ones([batch_size, grid_size**2], dtype=torch.float, device=device)
                / grid_size**2
            )
            loss = torch.zeros([1], device=device)
            loss_all = 0.0
            for step in range(seq_len):
                lkhd = torch.flatten(
                    model(
                        batch_x_grid[:, step, :, :, :],
                        (
                            batch_x_objs[:, step, :, :]
                            if batch_x_objs is not None
                            else None
                        ),
                        batch_x_mask[:, step, :] if batch_x_mask is not None else None,
                    ),
                    1,
                )  # Shape: (batch_size, grid_size * grid_size)
                lkhd = lkhd * (1 - lkhd_min) + lkhd_min
                new_priors = predict(
                    priors.view([batch_size, grid_size, grid_size])
                ).flatten(
                    1
                )  # Shape: (batch_size, grid_size * grid_size)
                new_priors = new_priors * lkhd
                new_priors = new_priors / new_priors.sum(1, keepdim=True)
                priors = new_priors
                with torch.no_grad():
                    loss_all += nll(priors.log(), batch_y[:, step]).item()
                if args.only_opt_last and step < seq_len - 1:
                    continue
                loss += nll(priors.log(), batch_y[:, step])
            opt.zero_grad()
            loss.backward()
            opt.step()

            avg_loss += loss.item()
            avg_loss_all += avg_loss

        model.eval()
        with torch.no_grad():
            avg_valid_loss = 0.0
            avg_valid_loss_all = 0.0
            priors = (
                torch.ones(
                    [num_seqs_valid, grid_size**2], dtype=torch.float, device=device
                )
                / grid_size**2
            )
            for step in range(seq_len):
                lkhd = model(
                    valid_x_grid[:, step, :, :, :],
                    valid_x_objs[:, step, :, :] if args.use_objs else None,
                    valid_x_mask[:, step, :] if args.use_objs else None,
                ).view([num_seqs_valid, grid_size**2])
                lkhd = lkhd * (1 - lkhd_min) + lkhd_min

                new_priors = predict(
                    priors.view([num_seqs_valid, grid_size, grid_size])
                ).view([num_seqs_valid, grid_size**2])
                new_priors = new_priors * lkhd
                new_priors = new_priors / new_priors.sum(1, keepdim=True)
                priors = new_priors
                with torch.no_grad():
                    avg_valid_loss_all += nll(priors.log(), valid_y[:, step]).item()
                if args.only_opt_last and step < seq_len - 1:
                    continue
                avg_valid_loss += nll(priors.log(), valid_y[:, step]).item()

        avg_loss = avg_loss / batches_per_epoch

        wandb.log(
            {
                "train_loss": avg_loss,
                "train_loss_all": avg_loss_all,
                "valid_loss": avg_valid_loss,
                "valid_loss_all": avg_valid_loss_all,
            }
        )

        if epoch % args.save_every == 0:
            save_model(model, str(chkpt_path / f"model-{epoch}.safetensors"))


if __name__ == "__main__":
    main()
