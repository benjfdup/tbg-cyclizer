### Overview: ###
### Basically does inference with the model. TBG full encodes information about atom topology in its running.
### Ultimately, I will use TBG full for training and inference.

### RUNS!!! Just is slow ###

### vvv IMPORTS vvv ###
import torch
import torchdyn.models as tdmls

import numpy as np

from bgflow import (
    DiffEqFlow,
    MeanFreeNormalDistribution,
)

from bgflow.utils import (
    as_numpy,
)

from tbg.models2 import EGNN_dynamics_transferable_MD
from bgflow import BlackBoxDynamics

import os
import tqdm
import mdtraj as md
import sys
### ^^^ IMPORTS ^^^ ###

data_path = "/home/bfd21/rds/hpc-work/2AA-complete" # use a dummy dataset, not the full one.
n_dimensions = 3 # SE(3), lets goooooo

directory = os.fsencode(data_path + "/val") # just loads validation data set
validation_peptides = [] # initializes list
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".pdb"): # loads into the validation set if & only if its the right file type (.proteinDataBase)
        validation_peptides.append(filename[:2]) # adds just the first 2 characters to the list (validation_peptides)

max_atom_number = 0 # instantiating a variable which will be used as a parameter of the EGNN used.
atom_dict = {"H": 0, "C": 1, "N": 2, "O": 3, "S": 4} # dictionary for atom numbers in Amino Acids.
scaling = 30 # scaling factor. Unsure exactly what this does, as its not referenced elsewhere

priors = {} # instantiating dictionary of prior distributions given the (di?-)peptide atom sequence.
# theoretically, priors dont really matter. But helps preserve SE(3) theoretical invariance.

topologies = {} # instantiating dictionary for loading in the mdtraj.topoligies of each peptide in the validation set.

atom_types_dict = {} # instantiating dictionary of atom types in each validation peptide

h_dict = {} # stores 1 hot encoding for each peptide -- does vector including topology information? not directly.

n_encodings = 5 # is this selected arbitratily?
for peptide in tqdm.tqdm(validation_peptides):

    topologies[peptide] = md.load_topology(
        data_path + f"/val/{peptide}-traj-state0.pdb" # precomputed topology from the relevant peptide structure
    ) # gets the topology of the relevant peptide, loads that into the relevant dictionary

    # I imagine getting the relevant topology will be non-trivial for my own example, given my (un-)familiarity with the software.

    atom_types = []
    n_atoms = len(list(topologies[peptide].atoms))
    for atom_name in topologies[peptide].atoms:
        atom_types.append(atom_name.name[0]) # appending the first character of each atom name -- perhaps all amin.acid. atom names
        # are one char (C, H, etc)
    atom_types_dict[peptide] = np.array(
        [atom_dict[atom_type] for atom_type in atom_types]
    )
    h_dict[peptide] = torch.nn.functional.one_hot(
        torch.tensor(atom_types_dict[peptide]), num_classes=n_encodings
    )
    priors[peptide] = MeanFreeNormalDistribution(
        n_atoms * n_dimensions, n_atoms, two_event_dims=False
    ).cuda()


directory = os.fsencode(data_path + "/train") # just adding more information to the topologies dictionary, etc.
validation_peptides = []
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".pdb"):
        validation_peptides.append(filename[:2])
for peptide in tqdm.tqdm(validation_peptides):

    topologies[peptide] = md.load_topology(
        data_path + f"/train/{peptide}-traj-state0.pdb"
    )
    atom_types = []
    n_atoms = len(list(topologies[peptide].atoms))
    for atom_name in topologies[peptide].atoms:
        atom_types.append(atom_name.name[0])
    atom_types_dict[peptide] = np.array(
        [atom_dict[atom_type] for atom_type in atom_types]
    )
    h_dict[peptide] = torch.nn.functional.one_hot(
        torch.tensor(atom_types_dict[peptide]), num_classes=n_encodings
    )
    priors[peptide] = MeanFreeNormalDistribution(
        n_atoms * n_dimensions, n_atoms, two_event_dims=False
    ).cuda()

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
    atom_types = []
    n_atoms = len(list(topologies[peptide].atoms))
    for atom_name in topologies[peptide].atoms:
        atom_types.append(atom_name.name[0])
    atom_types_dict[peptide] = np.array(
        [atom_dict[atom_type] for atom_type in atom_types]
    )
    h_dict[peptide] = torch.nn.functional.one_hot(
        torch.tensor(atom_types_dict[peptide]), num_classes=n_encodings
    )
    priors[peptide] = MeanFreeNormalDistribution(
        n_atoms * n_dimensions, n_atoms, two_event_dims=False
    ).cuda()

#for peptide in topologies:
#    n_atoms = len(list(topologies[peptide].atoms))  # Get the number of atoms for the peptide
#    if n_atoms > max_atom_number:
#        max_atom_number = n_atoms  # Update max_atom_number if this peptide has more atoms

max_atom_number = 51
# ultimately equivalent to the for loop above... which is thus unnecessary.

#peptide = sys.argv[1]
peptide = 'AA' #Normally need to pass an argument to this. here for testing...

### vvv NEED TO REVIEW BELOW CODE vvv -----------------------------------------------------------------------###

class BruteForceEstimatorFast(torch.nn.Module): # computes the divergence of the flow, so that we can invert it.
    """
    Exact bruteforce estimation of the divergence of a dynamics function.
    """

    def __init__(self):
        super().__init__()

    def forward(self, dynamics, t, xs):

        with torch.set_grad_enabled(True): ### I believe this is whats causing the slowness.
            xs.requires_grad_(True)
            x = [xs[:, [i]] for i in range(xs.size(1))]

            dxs = dynamics(t, torch.cat(x, dim=1)) ### this might be inducing slowness...

            assert len(dxs.shape) == 2, "`dxs` must have shape [n_btach, system_dim]"
            divergence = 0
            for i in range(xs.size(1)): ### I think this loop is causing the slowness...
                divergence += torch.autograd.grad( ###SLOW###
                    dxs[:, [i]], x[i], torch.ones_like(dxs[:, [i]]), retain_graph=True
                )[0]
        
        ## Add print here just as breakpoint?
        return dxs, -divergence.view(-1, 1) #whatever comes after this might be what is slow


net_dynamics = EGNN_dynamics_transferable_MD( # look at source of this
    n_particles=max_atom_number,
    h_size=n_encodings, # number of atoms
    device="cuda",
    n_dimension=n_dimensions,
    hidden_nf=128, # number of neurons per hidden layer
    act_fn=torch.nn.SiLU(), # activation function of the neurons
    n_layers=9,
    recurrent=True, # added recurrence (unsure exactly how, details are in the paper)
    tanh=True,
    attention=True, # not sure how attention is used here.
    condition_time=True,
    mode="egnn_dynamics",
    agg="sum", # aggregation function is the sum.
)

bb_dynamics = BlackBoxDynamics( # defines the dynamics to be integrated over.
    dynamics_function=net_dynamics, divergence_estimator=BruteForceEstimatorFast()
)

flow = DiffEqFlow(dynamics=bb_dynamics) # this is what ends up being slow, I think.
filename = "tbg"
PATH_last = f"/home/bfd21/rds/hpc-work/tbg/models/{filename}"
checkpoint = torch.load(PATH_last)
flow.load_state_dict(checkpoint["model_state_dict"])
loaded_epoch = checkpoint["epoch"]
global_it = checkpoint["global_it"]
print("Successfully loaded model")


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


#n_samples = 500
#n_sample_batches = 200

n_samples = 40 # REDUCED FOR TESTING
n_sample_batches = 1 # REDUCED FOR TESTING

dim = len(h_dict[peptide]) * 3
with_dlogp = False # default is true.

### ^^^ NEED TO REVIEW ABOVE CODE ^^^ -----------------------------------------------------------------------###

if with_dlogp:
    try:
        npz = np.load(f"/home/bfd21/rds/hpc-work/result_data/{filename}_{peptide}.npz")
        latent_np = npz["latent_np"]
        samples_np = npz["samples_np"]
        dlogp_np = npz["dlogp_np"]
        print("Successfully loaded samples")
    except:
        print("Start new sampling")
        latent_np = np.empty(shape=(0))
        samples_np = np.empty(shape=(0))
        # log_w_np = np.empty(shape=(0))
        dlogp_np = np.empty(shape=(0))
        # energies_np = np.empty(shape=(0))
        # distances_x_np = np.empty(shape=(0))
    print("Sampling with dlogp")
    print(peptide)
    
    for i in tqdm.tqdm(range(n_sample_batches)):
        with torch.no_grad():
            latent = priors[peptide].sample(n_samples)
            latent = torch.nn.functional.pad( # I believe this is fairly standard padding, for inference purposes
                latent, (0, (max_atom_number - len(h_dict[peptide])) * 3)
            )

            samples, dlogp = flow(latent) # This is the slow part... investigate thoroughly.

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
    n_samples *= 10
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

    node = tdmls.NeuralODE(
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
            traj = node.trajectory( ### ALSO RIDICULOUSLY SLOW LETS SEE WHY ###
                latent,
                t_span=t_span,
            )
            latent_np = np.append(latent_np, latent[:, :dim].detach().cpu().numpy())
            samples_np = np.append(samples_np, as_numpy(traj[-1])[:, :dim])
        np.savez( # Potentially got this to work... come back.
            f"/home/bfd21/rds/hpc-work/tbg/result_data/{filename}_{peptide}_nologp_TEST",
            latent_np=latent_np.reshape(-1, dim),
            samples_np=samples_np.reshape(-1, dim),
        )
