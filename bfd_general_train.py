### IMPORTS & SETUP ###
import torch
import numpy as np
import mdtraj as md

from bgflow.utils import (
    IndexBatchIterator,
)
from bgflow import (
    DiffEqFlow, # look into how this works
    MeanFreeNormalDistribution, # look into how this works, too.
)

from tbg.models2 import EGNN_dynamics_AD2_cat_bb_all_sc_adjacent
#from tbg.models2 import EGNN_dynamics_transferable_MD, EGNN_dynamics_AD2_cat
from bgflow import BlackBoxDynamics, BruteForceEstimator

import time
from tqdm import tqdm
from loguru import logger
import wandb

from bfd_conditionals import amino_dict, atom_types_ecoding
import os
import gc

### THINGS TO CHANGE vvv
log_dir = "/home/bfd21/rds/hpc-work/tbg/jobs/job-Dec-17" # where to store the training logs. Dont include slash at end
proj_name = "bb_all_sc_adj_N-Cap2"

pdb_path = "/home/bfd21/rds/hpc-work/sample_macrocycle_md/N-Cap2/system.pdb"
data_path = "/home/bfd21/rds/hpc-work/sample_macrocycle_md/N-Cap2/processed_train.npy"

# where to save the new model/load a previous model
PATH_last = "/home/bfd21/rds/hpc-work/tbg/bfd_models/Dec-17-2024/N-Cap2_bb_all_sc_adj.pth"

n_batch = 256 # batch size to be used. may need to play around with this if system runs out of memory
n_epochs = 100
### THINGS TO CHANGE ^^^

os.makedirs(f'{log_dir}/logs', exist_ok=True)

topology = md.load_topology(
        pdb_path # encodes the bond topology of the atoms encoded.
)
n_particles = len(list(topology.atoms)) # number of atoms in the given dipeptide.
n_dimensions = 3
dim = n_particles * n_dimensions

prior = MeanFreeNormalDistribution(dim, n_particles, two_event_dims=False).cuda()
prior_cpu = MeanFreeNormalDistribution(dim, n_particles, two_event_dims=False)
num_residues = len(list(topology.residues))


### INITIALIZING EMBEDDING OF PEPTIDE TO BE TRAINED ON vvv.
atom_types = []
amino_idx = []
amino_types = []

for i, amino in enumerate(topology.residues): # looping over the individual amino acids in the dipeptide.

    # data cleaning loop
    for atom_name in amino.atoms:
        amino_idx.append(i) # will end up being of shape (n, ), where n is the total number of atoms. Each index represents
            
        # number corresponding to aa name.
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
        if atom_name.name[:2] == "OE" or atom_name.name[:2] == "OD": # cleaning specific kinds of oxygen (per tbg paper)
            atom_name.name = atom_name.name[:-1]
        atom_types.append(atom_name.name)
    
atom_types = np.array([atom_types_ecoding[atom_type] for atom_type in atom_types])
# encodes information on the atom type, as well as its place in the amino acid
# encodes values 0 - 53, inclusive.

atom_onehot = torch.nn.functional.one_hot( # converts each atom type to a one hot encoding.
    torch.tensor(atom_types), num_classes=len(atom_types_ecoding) # 55 classes. changes with ecoding.
) # makes a tensor of shape (n, 54), where n is the number of atoms in the dipeptide, and 54 is the number of atom types.

amino_idx_onehot = torch.nn.functional.one_hot( # encodes the position as a 1-hot vector. Note that num_classes = 2 because
    torch.tensor(amino_idx), num_classes=num_residues
) # encodes the position as a 1-hot vector. Note that num_classes = 2 because we are only working with di-peptides here...

amino_types_onehot = torch.nn.functional.one_hot(
    torch.tensor(amino_types), num_classes=len(amino_dict)
) 
# one hot encoding of the amino acid type. 20 different kinds of amino acid,
# hence, 20 classes.
# of shape (n, 20), where n is the number of atoms and 20 is the number of amino acid classes here...

h_initial = torch.cat(
    [amino_idx_onehot, amino_types_onehot, atom_onehot], dim=1 # concatinates all of this information along the first axis
        # ie: along each atom. Therefore h_dict[peptide] is ultimately of shape: (n, 54 + 2 + 20)
        # or, put another way, h_dict[peptide] is of shape (n, len(atom_types_ecoding) + aa_length + #_of_amino_acids)
)

### INITIAL EMBEDDING COMPLETE (ALSO CALLED H) ^^^

# dynamics
brute_force_estimator = BruteForceEstimator()
net_dynamics = EGNN_dynamics_AD2_cat_bb_all_sc_adjacent( # DYNAMICS MODEL TO BE USED... YOU CAN CHANGE THIS.
    pdb_file=pdb_path,
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

#actual training:
data = np.load(data_path)
data_smaller = torch.from_numpy(data.reshape(data.shape[0], -1)).float()

del data  # Delete the large numpy array
gc.collect()  # Force garbage collection to free up RAM

batch_iter = IndexBatchIterator(len(data_smaller), n_batch)

optim = torch.optim.Adam(flow.parameters(), lr=5e-4)

### TRAINING IS BELOW HERE ###

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

sigma = 0.01 # std of datapoints for flowmatching

# Initialize loguru logger
logger.add(f"{log_dir}/training_log.log", rotation="500 MB", retention="10 days", level="INFO")

# Initialize wandb
wandb.init(
    project=proj_name,  # Name of your wandb project
    config={
        "n_particles": n_particles,
        "dim": dim,
        "hidden_nf": 64,
        "n_layers": 5,
        "n_epochs": n_epochs,
        "total_epochs": total_epochs,
        "batch_size": n_batch,
        "learning_rate": 5e-4,
        "sigma": sigma,
        "data_path": data_path,
        "model_dynamics": "EGNN_dynamics_AD2_cat_bb_all_sc_adj" ## CAN CHANGE THIS TOO
    },
    dir=f"{log_dir}/logs"
)

start_time = time.time()

rate_switch_triggered = False # bool to prevent the learning rate switching block from continually executing.print('Beginning training:')
for epoch in tqdm(range(n_epochs), desc="Epoch Progress", unit="epoch"):
    if (epoch + start_epoch >= 500) and not rate_switch_triggered:
        rate_switch_triggered = True
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

        # calculate regression loss
        mu_t = x0 * (1 - t) + x1 * t
        sigma_t = sigma
        noise = prior.sample(batchsize)
        x = mu_t + sigma_t * noise
        ut = x1 - x0
        vt = flow._dynamics._dynamics._dynamics_function(t, x) # be careful with too large batch sizes
        loss = torch.mean((vt - ut) ** 2)
        loss.backward()
        optim.step()

        epoch_loss += loss.item()
    
    if epoch % 20 == 0 or epoch == n_epochs - 1:
        print(epoch)

        avg_loss = epoch_loss / len(batch_iter)
        elapsed_time = time.time() - start_time

        # Log to wandb
        wandb.log({"Loss/train": avg_loss, "Epoch": epoch, "Time": elapsed_time})

        # Log to loguru
        logger.info(f"Epoch: {epoch}, Loss: {avg_loss:.6f}, Time: {elapsed_time:.2f}s")

        # Ensure the directory exists
        os.makedirs(os.path.dirname(PATH_last), exist_ok=True)

        torch.save(
            {
                "model_state_dict": flow.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "epoch": epoch,
            },
            PATH_last,
        )

# Save final model state
print("Training completed successfully.")

# Final logs and cleanup
logger.info("Training completed successfully.")
wandb.finish()