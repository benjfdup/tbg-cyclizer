### Imports ###

import torch
import numpy as np

from bgflow.utils import (
    as_numpy,
)
from bgflow import (
    DiffEqFlow,
    MeanFreeNormalDistribution,
)
from tbg.models2 import EGNN_dynamics_transferable_MD
from bgflow import BlackBoxDynamics

import os
import tqdm
import mdtraj as md
import sys

### Path Definitions ###

peptide = "KS"

data_path = "/home/bfd21/rds/hpc-work/2AA-complete"
ecoding_path = "/home/bfd21/rds/hpc-work/tbg/data/2AA-1-large"
n_dimensions = 3

directory = os.fsencode(data_path + "/val") # just the validation path, in bytes
validation_peptides = [] # list storing all validation peptides
for file in os.listdir(directory): # looping over all files in the validation directory
    filename = os.fsdecode(file)
    if filename.endswith(".pdb"):
        validation_peptides.append(filename[:2]) # if the file is of the protein database format, include filename in the validation list

max_atom_number = 0 # DK
atom_dict = {"H": 0, "C": 1, "N": 2, "O": 3, "S": 4} # atom type dictionary -- those avaliable to the model
scaling = 30  # DK

### Loading Validation Peptides ###

data_path = "/home/bfd21/rds/hpc-work/2AA-complete"
ecoding_path = "/home/bfd21/rds/hpc-work/tbg/data/2AA-1-large"
n_dimensions = 3

directory = os.fsencode(data_path + "/val")
validation_peptides = []
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".pdb"):
        validation_peptides.append(filename[:2])

max_atom_number = 0
atom_dict = {"H": 0, "C": 1, "N": 2, "O": 3, "S": 4}
scaling = 30

### Initialization of Dictionaries and Variables ###

priors = {}
topologies = {}
atom_types_dict = {}
h_dict = {}
n_encodings = 76
atom_types_ecoding = np.load(
    ecoding_path + "/atom_types_ecoding.npy", allow_pickle=True
).item()

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
}

### Processing Each Peptide and Creating One-Hot Encodings ###

for peptide in tqdm.tqdm(validation_peptides):

    topologies[peptide] = md.load_topology(
        data_path + f"/val/{peptide}-traj-state0.pdb"
    )
    n_atoms = len(list(topologies[peptide].atoms))
    
    atom_types = []
    amino_idx = []
    amino_types = []

    for i, amino in enumerate(topologies[peptide].residues):
        for atom_name in amino.atoms:
            amino_idx.append(i)
            amino_types.append(amino_dict[amino.name])

            # Handling different atom names
            if atom_name.name[0] == "H" and atom_name.name[-1] in ("1", "2", "3"):
                if amino_dict[amino.name] in (8, 13, 17, 18) and atom_name.name[:2] in ("HE", "HD", "HZ", "HH"):
                    pass
                else:
                    atom_name.name = atom_name.name[:-1]

            if atom_name.name[:2] == "OE" or atom_name.name[:2] == "OD":
                atom_name.name = atom_name.name[:-1]

            atom_types.append(atom_name.name)
    
    # Encoding atoms and amino acids
    atom_types_dict[peptide] = np.array([atom_types_ecoding[atom_type] for atom_type in atom_types])
    
    atom_onehot = torch.nn.functional.one_hot(torch.tensor(atom_types_dict[peptide]), num_classes=len(atom_types_ecoding))
    amino_idx_onehot = torch.nn.functional.one_hot(torch.tensor(amino_idx), num_classes=2)
    amino_types_onehot = torch.nn.functional.one_hot(torch.tensor(amino_types), num_classes=20)

    # Concatenating encodings
    h_dict[peptide] = torch.cat([amino_idx_onehot, amino_types_onehot, atom_onehot], dim=1)

    # Setting up priors
    priors[peptide] = MeanFreeNormalDistribution(n_atoms * n_dimensions, n_atoms, two_event_dims=False).cuda()

### Train Peptides (Similar Logic as Validation Peptides) ###

directory = os.fsencode(data_path + "/train")
validation_peptides = []
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".pdb"):
        validation_peptides.append(filename[:2])
for peptide in tqdm.tqdm(validation_peptides):

    topologies[peptide] = md.load_topology(
        data_path + f"/train/{peptide}-traj-state0.pdb"
    )
    n_atoms = len(list(topologies[peptide].atoms))
    atom_types = []
    atom_types = []
    amino_idx = []
    amino_types = []
    for i, amino in enumerate(topologies[peptide].residues):

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
    atom_types_dict[peptide] = np.array(
        [atom_types_ecoding[atom_type] for atom_type in atom_types]
    )
    atom_onehot = torch.nn.functional.one_hot(
        torch.tensor(atom_types_dict[peptide]), num_classes=len(atom_types_ecoding)
    )
    amino_idx_onehot = torch.nn.functional.one_hot(
        torch.tensor(amino_idx), num_classes=2
    )
    amino_types_onehot = torch.nn.functional.one_hot(
        torch.tensor(amino_types), num_classes=20
    )

    h_dict[peptide] = torch.cat(
        [amino_idx_onehot, amino_types_onehot, atom_onehot], dim=1
    )
    priors[peptide] = MeanFreeNormalDistribution(
        n_atoms * n_dimensions, n_atoms, two_event_dims=False
    ).cuda()

### Test Peptides (Similar Logic as Validation Peptides) ###

directory = os.fsencode(data_path + "/test")
validation_peptides = []
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".pdb"):
        validation_peptides.append(filename[:2])
for peptide in tqdm.tqdm(validation_peptides):

    topologies[peptide] = md.load_topology(
        data_path + f"/test/{peptide}-traj-state0.pdb"
    )
    n_atoms = len(list(topologies[peptide].atoms))
    atom_types = []
    atom_types = []
    amino_idx = []
    amino_types = []
    for i, amino in enumerate(topologies[peptide].residues):

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
    atom_types_dict[peptide] = np.array(
        [atom_types_ecoding[atom_type] for atom_type in atom_types]
    )
    atom_onehot = torch.nn.functional.one_hot(
        torch.tensor(atom_types_dict[peptide]), num_classes=len(atom_types_ecoding)
    )
    amino_idx_onehot = torch.nn.functional.one_hot(
        torch.tensor(amino_idx), num_classes=2
    )
    amino_types_onehot = torch.nn.functional.one_hot(
        torch.tensor(amino_types), num_classes=20
    )

    h_dict[peptide] = torch.cat(
        [amino_idx_onehot, amino_types_onehot, atom_onehot], dim=1
    )
    priors[peptide] = MeanFreeNormalDistribution(
        n_atoms * n_dimensions, n_atoms, two_event_dims=False
    ).cuda()

### Defining and Loading the EGNN Dynamics with Brute Force Estimation ###

max_atom_number = 51

peptide = "KS"


class BruteForceEstimatorFast(torch.nn.Module):
    """
    Exact bruteforce estimation of the divergence of a dynamics function.
    """

    def __init__(self):
        super().__init__()

    def forward(self, dynamics, t, xs):

        with torch.set_grad_enabled(True):
            xs.requires_grad_(True)
            x = [xs[:, [i]] for i in range(xs.size(1))]

            dxs = dynamics(t, torch.cat(x, dim=1))

            assert len(dxs.shape) == 2, "`dxs` must have shape [n_btach, system_dim]"
            divergence = 0
            for i in range(xs.size(1)):
                divergence += torch.autograd.grad(
                    dxs[:, [i]], x[i], torch.ones_like(dxs[:, [i]]), retain_graph=True
                )[0]

        return dxs, -divergence.view(-1, 1)

hidden_nf = 128
n_layers = 9
net_dynamics = EGNN_dynamics_transferable_MD(
    n_particles=max_atom_number,
    h_size=n_encodings,
    device="cuda",
    n_dimension=n_dimensions,
    hidden_nf=hidden_nf,
    act_fn=torch.nn.SiLU(),
    n_layers=n_layers,
    recurrent=True,
    tanh=True,
    attention=True,
    condition_time=True,
    mode="egnn_dynamics",
    agg="sum",
)

bb_dynamics = BlackBoxDynamics(
    dynamics_function=net_dynamics, divergence_estimator=BruteForceEstimatorFast()
)

flow = DiffEqFlow(dynamics=bb_dynamics)

filename = f"tbg_full"

PATH_last = f"/home/bfd21/rds/hpc-work/tbg/models/{filename}"
checkpoint = torch.load(PATH_last)
flow.load_state_dict(checkpoint["model_state_dict"])
loaded_epoch = checkpoint["epoch"]
global_it = checkpoint["global_it"]
print(f"Successfully loaded model {filename}")

### Wrapper for Masking and Handling Particles in EGNN Dynamics ###

class NetDynamicsWrapper(torch.nn.Module):
    def __init__(self, net_dynamics, n_particles, max_n_particles, h_initial):
        super().__init__()
        self.net_dynamics = net_dynamics
        self.n_particles = n_particles
        mask = torch.ones((1, n_particles))
        mask = torch.nn.functional.pad(
            mask, (0, (max_n_particles - n_particles))
        )  # .bool()
        edge_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
        # mask diagonal
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
        edge_mask *= diag_mask
        self.node_mask = mask
        self.edge_mask = edge_mask
        self.h_initial = torch.cat(
            [h_initial, torch.zeros(max_n_particles - n_particles, h_initial.size(1))]
        ).unsqueeze(0)

    def forward(self, t, xs, args=None):
        n_batch = xs.size(0)
        node_mask = self.node_mask.repeat(n_batch, 1).to(xs)
        edge_mask = self.edge_mask.repeat(n_batch, 1, 1).to(xs)
        h_initial = self.h_initial.repeat(n_batch, 1, 1).to(xs)
        return self.net_dynamics(
            t, xs, h_initial, node_mask=node_mask, edge_mask=edge_mask
        )


net_dynamics_wrapper = NetDynamicsWrapper(
    net_dynamics,
    n_particles=len(h_dict[peptide]),
    max_n_particles=max_atom_number,
    h_initial=h_dict[peptide],
)


flow._dynamics._dynamics._dynamics_function = net_dynamics_wrapper
flow._integrator_atol = 1e-4
flow._integrator_rtol = 1e-4
flow._use_checkpoints = False
flow._kwargs = {}

### Setup Variables and Parameters ###
n_samples = 500
n_sample_batches = 200

dim = len(h_dict[peptide]) * 3  # Peptide dimension (3D)
with_dlogp = True  # Flag for sampling with dlogp (log probability)


### Loading Precomputed Samples (if available) ###

if with_dlogp:
    try:
        npz = np.load(f"result_data/{filename}_{peptide}.npz")
        latent_np = npz["latent_np"]
        samples_np = npz["samples_np"]
        dlogp_np = npz["dlogp_np"]
        print("Successfully loaded samples")
    except:
        print("Start new sampling")
        latent_np = np.empty(shape=(0))
        samples_np = np.empty(shape=(0))
        dlogp_np = np.empty(shape=(0))

    print("Sampling with dlogp")
    print(peptide)
    for i in tqdm.tqdm(range(n_sample_batches)):
        with torch.no_grad():
            latent = priors[peptide].sample(n_samples)
            latent = torch.nn.functional.pad(
                latent, (0, (max_atom_number - len(h_dict[peptide])) * 3)
            )

            samples, dlogp = flow(latent)

            latent_np = np.append(latent_np, latent[:, :dim].detach().cpu().numpy())
            samples_np = np.append(samples_np, samples[:, :dim].detach().cpu().numpy())

            dlogp_np = np.append(dlogp_np, as_numpy(dlogp))

        # print(i)
        np.savez(
            f"result_data/{filename}_{peptide}",
            latent_np=latent_np.reshape(-1, dim),
            samples_np=samples_np.reshape(-1, dim),
            dlogp_np=dlogp_np,
        )
else:
    n_samples *= 10 # Increase number of samples if not using dlogp
    try:
        npz = np.load(f"result_data/{filename}_{peptide}_nologp.npz")
        latent_np = npz["latent_np"]
        samples_np = npz["samples_np"]
        print("Successfully loaded samples")
    except:
        print("Start new sampling")
        latent_np = np.empty(shape=(0))
        samples_np = np.empty(shape=(0))
    print("Sampling without dlogp")
    from torchdyn.core import NeuralODE

    node = NeuralODE(
        net_dynamics_wrapper,
        solver="dopri5",
        sensitivity="adjoint",
        atol=1e-4,
        rtol=1e-4,
    )
    t_span = torch.linspace(0, 1, 100)
    for i in tqdm.tqdm(range(n_sample_batches)):
        with torch.no_grad():
            latent = priors[peptide].sample(n_samples)
            latent = torch.nn.functional.pad(
                latent, (0, (max_atom_number - len(h_dict[peptide])) * 3)
            )
            traj = node.trajectory(
                latent,
                t_span=t_span,
            )
            latent_np = np.append(latent_np, latent[:, :dim].detach().cpu().numpy())
            samples_np = np.append(samples_np, as_numpy(traj[-1])[:, :dim])
        np.savez(
            f"result_data/{filename}_{peptide}_nologp",
            latent_np=latent_np.reshape(-1, dim),
            samples_np=samples_np.reshape(-1, dim),
        )