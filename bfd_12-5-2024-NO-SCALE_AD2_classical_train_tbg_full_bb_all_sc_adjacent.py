### IMPORTS & SETUP ###
import torch
import numpy as np
import mdtraj as md

from bgflow.utils import (
    IndexBatchIterator,
)
from bgflow import (
    DiffEqFlow,
    MeanFreeNormalDistribution,
)
from tbg.models2 import EGNN_dynamics_AD2_cat_bb_all_sc_adjacent
from bgflow import BlackBoxDynamics, BruteForceEstimator

import time
from tqdm import tqdm
from loguru import logger
import wandb
import os

from bfd_constants import *

os.makedirs("/home/bfd21/rds/hpc-work/tbg/jobs/job-Nov-28/logs", exist_ok=True)

# Setup
n_particles = 177
n_dimensions = 3
dim = n_particles * n_dimensions
prior = MeanFreeNormalDistribution(dim, n_particles, two_event_dims=False).cuda()
prior_cpu = MeanFreeNormalDistribution(dim, n_particles, two_event_dims=False)
scaling = 30

# Load topology
topology = md.load_topology(
    "/home/bfd21/rds/hpc-work/sample_cyclic_md/ligand-only/dummy1/l1.pdb"
)
num_residues = len(list(topology.residues))
n_particles = len(list(topology.atoms))

atom_types = []
amino_idx = []
amino_types = []

for i, amino in enumerate(topology.residues):
    for atom_name in amino.atoms:
        amino_idx.append(i)
        amino_types.append(amino_dict[amino.name])
        if atom_name.name[0] == "H" and atom_name.name[-1] in ("1", "2", "3"):
            if amino_dict[amino.name] in (8, 13, 17, 18) and atom_name.name[:2] in (
                "HE",
                "HD",
                "HZ",
                "HH",
            ):
                pass
            else:
                atom_name.name = atom_name.name[:-1]
        if atom_name.name[:2] == "OE" or atom_name.name[:2] == "OD":
            atom_name.name = atom_name.name[:-1]
        atom_types.append(atom_name.name)

atom_types = np.array([atom_types_ecoding[atom_type] for atom_type in atom_types])
atom_onehot = torch.nn.functional.one_hot(
    torch.tensor(atom_types), num_classes=len(atom_types_ecoding)
)
amino_idx_onehot = torch.nn.functional.one_hot(
    torch.tensor(amino_idx), num_classes=num_residues
)
amino_types_onehot = torch.nn.functional.one_hot(
    torch.tensor(amino_types), num_classes=len(amino_dict)
)

h_initial = torch.cat(
    [amino_idx_onehot, amino_types_onehot, atom_onehot], dim=1
)

# Dynamics setup
brute_force_estimator = BruteForceEstimator()
net_dynamics = EGNN_dynamics_AD2_cat_bb_all_sc_adjacent(
    pdb_file="/home/bfd21/rds/hpc-work/sample_cyclic_md/ligand-only/dummy1/l1.pdb",
    n_particles=n_particles,
    device="cuda",
    n_dimension=dim // n_particles,
    h_initial=h_initial,
    hidden_nf=64,
    act_fn=torch.nn.SiLU(),
    n_layers=5,
    recurrent=True,
    tanh=True,
    attention=True,
    condition_time=True,
    mode="egnn_dynamics",
    agg="sum",
)

bb_dynamics = BlackBoxDynamics(
    dynamics_function=net_dynamics, divergence_estimator=brute_force_estimator
)
flow = DiffEqFlow(dynamics=bb_dynamics)

# Training setup
n_batch = 256
data_path = "/home/bfd21/rds/hpc-work/sample_cyclic_md/ligand-only/dummy1/dummy1_train.npy"
data_smaller = torch.from_numpy(np.load(data_path)).float()
batch_iter = IndexBatchIterator(len(data_smaller), n_batch)

optim = torch.optim.Adam(flow.parameters(), lr=5e-4)
n_epochs = 100
PATH_last = "/home/bfd21/rds/hpc-work/tbg/bfd_models/Nov-28-2024/Nov-28-2024-Flow-Matching-L1-bb_all_sc_adjacent.pth"

# Load checkpoint if exists
if os.path.exists(PATH_last):
    checkpoint = torch.load(PATH_last)
    flow.load_state_dict(checkpoint["model_state_dict"])
    optim.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    print(f"Resuming training from epoch {start_epoch}")
else:
    start_epoch = 0
    print("No checkpoint found. Starting training from scratch.")

total_epochs = start_epoch + n_epochs

# Initialize logger
logger.add("/home/bfd21/rds/hpc-work/tbg/jobs/job-Nov-28/training_log.log", rotation="500 MB", retention="10 days", level="INFO")

# Initialize wandb
wandb.init(
    project="L1-Nov-28-2024-bb_all_sc_adj",
    config={
        "n_particles": n_particles,
        "dim": dim,
        "hidden_nf": 64,
        "n_layers": 5,
        "total_epochs": total_epochs,
        "batch_size": n_batch,
        "learning_rate": 5e-4,
        "sigma": 0.01,
        "data_path": data_path,
        "model_dynamics": "EGNN_dynamics_AD2_cat_bb_all_sc_adjacent",
        "resume_training": os.path.exists(PATH_last),
    },
    dir="/home/bfd21/rds/hpc-work/tbg/jobs/job-Nov-28/logs",
    resume="allow",
)

start_time = time.time()

print("Training:")
for epoch in tqdm(range(start_epoch, total_epochs), desc="Epoch Progress", unit="epoch"):
    if epoch == 500:
        for g in optim.param_groups:
            g["lr"] = 5e-5
        logger.info("Learning rate updated to 5e-5 at epoch 500.")

    epoch_loss = 0
    for it, idx in enumerate(batch_iter):
        optim.zero_grad()

        x1 = data_smaller[idx].cuda()
        batchsize = x1.shape[0]

        t = torch.rand(batchsize, 1).cuda()
        x0 = prior_cpu.sample(batchsize).cuda()

        mu_t = x0 * (1 - t) + x1 * t
        sigma_t = 0.01
        noise = prior.sample(batchsize)
        x = mu_t + sigma_t * noise
        ut = x1 - x0
        vt = flow._dynamics._dynamics._dynamics_function(t, x)
        loss = torch.mean((vt - ut) ** 2)
        loss.backward()
        optim.step()

        epoch_loss += loss.item()

    if epoch % 100 == 0 or epoch == total_epochs - 1:
        avg_loss = epoch_loss / len(batch_iter)
        elapsed_time = time.time() - start_time

        wandb.log({"Loss/train": avg_loss, "Epoch": epoch, "Time": elapsed_time})
        logger.info(f"Epoch: {epoch}, Loss: {avg_loss:.6f}, Time: {elapsed_time:.2f}s")

        torch.save(
            {
                "model_state_dict": flow.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "epoch": epoch,
            },
            "/home/bfd21/rds/hpc-work/tbg/bfd_models/Nov-28-2024/L1-bb_all_sc_adjacent_NEW-12-5-24.pth",
        )

print("Training completed successfully.")
logger.info("Training completed successfully.")
wandb.finish()
