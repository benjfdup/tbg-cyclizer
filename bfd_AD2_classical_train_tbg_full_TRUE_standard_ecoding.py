import torch
import numpy as np

from bgflow.utils import IndexBatchIterator
from bgflow import DiffEqFlow, MeanFreeNormalDistribution
from tbg.models2 import EGNN_dynamics_AD2_cat
from bgflow import BlackBoxDynamics, BruteForceEstimator

from tqdm import tqdm
from loguru import logger
import wandb

import os
os.makedirs("/home/bfd21/rds/hpc-work/tbg/jobs/job-Nov-22_true_dialanine_full_standard_ecoding/logs", exist_ok=True)

n_particles = 22
n_dimensions = 3
dim = n_particles * n_dimensions

# atom types for backbone
atom_types = np.arange(22)
atom_types[[1, 2, 3]] = 2
atom_types[[19, 20, 21]] = 20
atom_types[[11, 12, 13]] = 12
h_initial = torch.nn.functional.one_hot(torch.tensor(atom_types))


# now set up a prior
prior = MeanFreeNormalDistribution(dim, n_particles, two_event_dims=False).cuda()
prior_cpu = MeanFreeNormalDistribution(dim, n_particles, two_event_dims=False)

brute_force_estimator = BruteForceEstimator()
net_dynamics = EGNN_dynamics_AD2_cat(
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

n_batch = 40
data_path = "/home/bfd21/rds/hpc-work/tbg/data/AD2/AD2_weighted.npy"
data_smaller = torch.from_numpy(np.load(data_path)).float()
batch_iter = IndexBatchIterator(len(data_smaller), n_batch)

optim = torch.optim.Adam(flow.parameters(), lr=5e-4)

n_epochs = 1000

PATH_last = "/home/bfd21/rds/hpc-work/tbg/bfd_models/Flow-Matching-AD2-amber-weighted-encoding-epoch-test.pth"

sigma = 0.01

# Initialize loguru logger
logger.add("/home/bfd21/rds/hpc-work/tbg/jobs/job-Nov-22_true_dialanine_full_standard_ecoding/training_log.log", rotation="500 MB", retention="10 days", level="INFO")

# Initialize wandb
wandb.init(
    project="Flow-Matching-AD2-classic-test_FULL_STANDARD_ECODING",  # Name of your wandb project
    config={
        "n_particles": n_particles,
        "dim": dim,
        "hidden_nf": 64,
        "n_layers": 5,
        "n_epochs": n_epochs,
        "batch_size": n_batch,
        "learning_rate": 5e-4,
        "sigma": sigma,
        "data_path": data_path,
        "model_dynamics": "EGNN_dynamics_AD2_cat"
    },
    dir="/home/bfd21/rds/hpc-work/tbg/jobs/job-Nov-22_true_dialanine_full_standard_ecoding/logs"
)

print('starting training!!')
for epoch in tqdm(range(n_epochs), desc="Epoch Progress", unit="epoch"):
    if epoch == 500:
        for g in optim.param_groups:
            g["lr"] = 5e-5

    for it, idx in enumerate(batch_iter):
        optim.zero_grad()

        x1 = data_smaller[idx].cuda()
        batchsize = x1.shape[0]

        t = torch.rand(batchsize, 1).cuda()
        x0 = prior_cpu.sample(batchsize).cuda()

        # calculate regression loss
        mu_t = x0 * (1 - t) + x1 * t
        sigma_t = sigma
        noise = prior.sample(batchsize)
        x = mu_t + sigma_t * noise
        ut = x1 - x0
        vt = flow._dynamics._dynamics._dynamics_function(t, x)
        loss = torch.mean((vt - ut) ** 2)
        loss.backward()
        optim.step()
    if epoch % 100 == 0:
        print(epoch)
        torch.save(
            {
                "model_state_dict": flow.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "epoch": epoch,
            },
            PATH_last,
        )

print('Training Completed Successfully')
