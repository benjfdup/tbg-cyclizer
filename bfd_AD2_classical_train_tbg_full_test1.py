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
from tbg.models2 import EGNN_dynamics_AD2_cat
#from tbg.models2 import EGNN_dynamics_transferable_MD, EGNN_dynamics_AD2_cat
from bgflow import BlackBoxDynamics, BruteForceEstimator

import time
from tqdm import tqdm
from loguru import logger
import wandb

import os
os.makedirs("/home/bfd21/rds/hpc-work/tbg/jobs/job-Nov-21-debug/logs", exist_ok=True)

### NOT DESIGNED TO TRAIN THE MODEL ON DIALANINE, BUT RATHER L1
# atom types for backbone, directory and setup information

n_particles = 177 # particles in ligand 1
n_dimensions = 3
dim = n_particles * n_dimensions

atom_types_ecoding = np.load(
    "/home/bfd21/rds/hpc-work/sample_cyclic_md/atom_types_ecoding.npy", allow_pickle=True
).item()

atom_types_ecoding = {
    'C': 0, 
    'CA': 1, 
    'CB': 2, 
    'CD': 3, 
    'CD1': 4, 
    'CD2': 5, 
    'CE': 6, 
    'CE1': 7, 
    'CE2': 8, 
    'CE3': 9, 
    'CG': 10, 
    'CG1': 11, 
    'CG2': 12, 
    'CH2': 13,
    'CL1': 54, # This is chlorine. Idk... Feels like a hacky solution
    'CZ': 14, 
    'CZ2': 15, 
    'CZ3': 16, 
    'H': 17, 
    'HA': 18, 
    'HB': 19, 
    'HD': 20, 
    'HD1': 21, 
    'HD2': 22, 
    'HE': 23, 
    'HE1': 24, 
    'HE2': 25, 
    'HE3': 26, 
    'HG': 27, 
    'HG1': 28, 
    'HG2': 29, 
    'HH': 30, 
    'HH1': 31, 
    'HH2': 32, 
    'HZ': 33, 
    'HZ2': 34, 
    'HZ3': 35, 
    'N': 36, 
    'ND1': 37, 
    'ND2': 38, 
    'NE': 39, 
    'NE1': 40, 
    'NE2': 41, 
    'NH1': 42, 
    'NH2': 43, 
    'NZ': 44, 
    'O': 45, 
    'OD': 46, 
    'OE': 47, 
    'OG': 48, 
    'OG1': 49, 
    'OH': 50, 
    'OXT': 51, 
    'SD': 52, 
    'SG': 53,
    }
# need to add CL1 to this??

amino_dict = {
    "ALA": 0,
    "ARG": 1,
    "ASN": 2,
    "ASP": 3,
    "CYS": 4,
    "GLN": 5,
    "GLU": 6,
    "GLY": 7,
    "HIS": 8,
    "ILE": 9,
    "LEU": 10,
    "LYS": 11,
    "MET": 12,
    "PHE": 13,
    "PRO": 14,
    "SER": 15,
    "THR": 16,
    "TRP": 17,
    "TYR": 18,
    "VAL": 19,
    "UNK": 20, # need to have a better way of handling this in the future...
}
atom_dict = {"H": 0, "C": 1, "N": 2, "O": 3, "S": 4}

prior = MeanFreeNormalDistribution(dim, n_particles, two_event_dims=False).cuda()
prior_cpu = MeanFreeNormalDistribution(dim, n_particles, two_event_dims=False)
scaling = 30 # scaling factor used to compute how distances are scaled from the CoM

topology = md.load_topology(
        "/home/bfd21/rds/hpc-work/sample_cyclic_md/ligand-only/dummy1/l1.pdb" # encodes the bond topology of the atoms encoded.
)

# Count the number of residues in the topology
num_residues = len(list(topology.residues))

n_particles = len(list(topology.atoms)) # number of atoms in the given dipeptide. Should be 177

atom_types = []
amino_idx = []
amino_types = []

for i, amino in enumerate(topology.residues): # looping over the individual amino acids in the dipeptide.

    # data cleaning loop
    for atom_name in amino.atoms:
        amino_idx.append(i) # will end up being of shape (n, ), where n is the total number of atoms. Each index represents
            
        # whether the atom is in the first or second amino acid of the dipeptide.
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

# dynamics
brute_force_estimator = BruteForceEstimator()
net_dynamics = EGNN_dynamics_AD2_cat( # maybe switch this out for some other netdynamics.
    # like EGNN_dynamics_transferable_MD
    # Lets just... see if this works (lol)
    # I would like to poke around in the EGNN_dynamics_AD2_cat code to see what makes it different
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
#n_batch = 256
n_batch = 40 # see if this runs out of memory.
data_path = "/home/bfd21/rds/hpc-work/sample_cyclic_md/ligand-only/dummy1/dummy1_train.npy"
data_smaller = torch.from_numpy(np.load(data_path)).float() # data will be loaded on-line I suppose...
batch_iter = IndexBatchIterator(len(data_smaller), n_batch)

optim = torch.optim.Adam(flow.parameters(), lr=5e-4)

n_epochs = 1000

#PATH_last = "models/Flow-Matching-AD2-amber-weighted-encoding" # TBG original
PATH_last = "/home/bfd21/rds/hpc-work/tbg/bfd_models/Flow-Matching-L1-test.pth"

# let's just see if this works :)

### TRAINING IS BELOW HERE ###
sigma = 0.01 # std of datapoints for flowmatching

# Initialize loguru logger
logger.add("/home/bfd21/rds/hpc-work/tbg/jobs/job-Nov-21-debug/training_log.log", rotation="500 MB", retention="10 days", level="INFO")

# Initialize wandb
wandb.init(
    project="Flow-Matching-L1-TEST-Nov-21-2024",  # Name of your wandb project
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
    dir="/home/bfd21/rds/hpc-work/tbg/jobs/job-Nov-21-debug/logs"
)

start_time = time.time()

print('Beginning training:')
for epoch in tqdm(range(n_epochs), desc="Epoch Progress", unit="epoch"):
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

        # calculate regression loss
        mu_t = x0 * (1 - t) + x1 * t
        sigma_t = sigma
        noise = prior.sample(batchsize)
        x = mu_t + sigma_t * noise
        ut = x1 - x0
        vt = flow._dynamics._dynamics._dynamics_function(t, x) # this is where the model runs out of memory.
        # why is this one line so demanding spatially..? Batch size too big? ### COME BACK HERE...
        loss = torch.mean((vt - ut) ** 2)
        loss.backward()
        optim.step()

        epoch_loss += loss.item()
    
    if epoch % 100 == 0 or epoch == n_epochs - 1:
        print(epoch)

        avg_loss = epoch_loss / len(batch_iter)
        elapsed_time = time.time() - start_time

        # Log to wandb
        wandb.log({"Loss/train": avg_loss, "Epoch": epoch, "Time": elapsed_time})

        # Log to loguru
        logger.info(f"Epoch: {epoch}, Loss: {avg_loss:.6f}, Time: {elapsed_time:.2f}s")

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