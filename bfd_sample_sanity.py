###########################################
# The intent of this file is simply to do one sanity check
# otherwise it is identical to bfd_general_sample.py
###########################################

import tqdm
import torch
import numpy as np
import mdtraj as md
import os
import pickle

from bgflow.utils import as_numpy
from bgflow import DiffEqFlow, BoltzmannGenerator, MeanFreeNormalDistribution, BlackBoxDynamics, BruteForceEstimator
from tbg.models2 import EGNN_dynamics_AD2_cat_bb_all_sc_adj_cyclic, EGNN_dynamics_AD2_cat_bb_all_sc_adjacent

from bfd_conditionals import cyclization_loss_handler, gaussian_w_t
from bfd_constants import *

### vvv -----===== THINGS TO CHANGE =====----- vvv
pdb_path = "/home/bfd21/rds/hpc-work/sample_macrocycle_md/N-Cap2/system.pdb"

filename = "N-Cap2_bb_all_sc_adj.pth" # model to be used for inference
PATH_last = f"/home/bfd21/rds/hpc-work/tbg/bfd_models/Dec-17-2024/{filename}" # path to model dir

save_dir = "/home/bfd21/rds/hpc-work/tbg/result_data/Jan-9-2025/"

if save_dir[-1] != "/": # DON'T CHANGE
    save_dir += "/" # DON'T CHANGE

save_data_name = "N-Cap2_bb_all_sc_adj_jan_9_samples_conditional" # DO NOT INCLUDE .npz extension here...

strategies = ['special', 'disulfide']
#['disulfide', 'amide', 'side_chain_amide', 'thioether', 'ester', 'hydrazone', 'h2t']

with_dlogp = False
### ^^^ -----===== THINGS TO CHANGE =====----- ^^^

# Extract the directory part from the template
save_dir_path = os.path.dirname(save_dir)

# Ensure the directory exists
os.makedirs(save_dir_path, exist_ok=True)

topology = md.load_topology(pdb_path) # encodes the bond topology of the atoms encoded.

# Count the number of residues in the topology
num_residues = len(list(topology.residues))

n_particles = len(list(topology.atoms)) # number of atoms in the given dipeptide. Should be 177(?)
n_dimensions = 3
dim = n_particles * n_dimensions

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

# now set up a prior
prior = MeanFreeNormalDistribution(dim, n_particles, two_event_dims=False).cuda() ### might this be causing the problems?
prior_cpu = MeanFreeNormalDistribution(dim, n_particles, two_event_dims=False)

# Initialize the cyclization loss function
#cyclization_loss_fn = initialize_cyclization_loss(
#    pdb_path="/path/to/your/pdb/file.pdb",
#    strategies=["disulfide", "amide", "h2t"],  # Add relevant strategies
#    alpha=-10  # Adjust alpha as needed
#)

brute_force_estimator = BruteForceEstimator()
#commented out is model for unconditional sampling.
#net_dynamics = EGNN_dynamics_AD2_cat_bb_all_sc_adjacent( ### CHANGE MODEL TO WHATEVER IS NECESSARY...
#    pdb_file=pdb_path,
#    n_particles=n_particles,
#    device="cuda",
#    n_dimension=dim // n_particles,
#    h_initial=h_initial,
#    hidden_nf=64,
#    act_fn=torch.nn.SiLU(),
#    n_layers=5,
#    recurrent=True,
#    tanh=True,
#    attention=True,
#    condition_time=True,
#    mode="egnn_dynamics",
#    agg="sum",
#)

loss_handler = cyclization_loss_handler(pdb_path = pdb_path,
                                        strategies=strategies,
                                        alpha = -0.5,
                                        )

w_t = gaussian_w_t(mu=0.5, s=0.1)

net_dynamics = EGNN_dynamics_AD2_cat_bb_all_sc_adj_cyclic( ### CHANGE MODEL TO WHATEVER IS NECESSARY...
    ### This might not work in this context, but lets just try it
    with_dlogp=with_dlogp,
    pdb_file=pdb_path,
    w_t=w_t,
    l_cyclic=loss_handler.compute_loss,
    #l_cyclic= lambda x: (0.01 * x ** 2).sum(dim=(1, 2)),
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

### TODO:
### Size of dt (& hence number of steps) seems to be determined by the error of the system... Try to verify this & see if you can fix
### the error below!!!

bb_dynamics = BlackBoxDynamics(
    dynamics_function=net_dynamics, divergence_estimator=brute_force_estimator
)

flow = DiffEqFlow(dynamics=bb_dynamics)

bg = BoltzmannGenerator(prior, flow, prior).cuda()

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

            assert len(dxs.shape) == 2, f"`dxs` must have shape [n_batch, system_dim]."
            divergence = 0
            for i in range(xs.size(1)):
                divergence += torch.autograd.grad(
                    dxs[:, [i]], x[i], torch.ones_like(dxs[:, [i]]), retain_graph=True
                )[0]

        return dxs, -divergence.view(-1, 1)


brute_force_estimator_fast = BruteForceEstimatorFast()
# use OTD in the evaluation process
bb_dynamics._divergence_estimator = brute_force_estimator_fast
bg.flow._integrator_atol = 1e-4
bg.flow._integrator_rtol = 1e-4
flow._use_checkpoints = False  ### INTERESTING TO PLAY AROUND WITH THIS, NO?
flow._kwargs = {}

checkpoint = torch.load(PATH_last)
flow.load_state_dict(checkpoint["model_state_dict"])

n_samples = 20 #10 #45 #400
n_sample_batches = 2 #500
latent_np = np.empty(shape=(0))
samples_np = np.empty(shape=(0))
dlogp_np = np.empty(shape=(0))
print(f"""
      -------======= START SAMPLING WITH {filename} =======-------
      """)

for i in tqdm.tqdm(range(n_sample_batches)):
    with torch.no_grad():
        if with_dlogp:
            samples, latent, dlogp = bg.sample(n_samples, with_latent=True, with_dlogp=with_dlogp) # with_dlogp=False for now
        else:
            samples, latent = bg.sample(n_samples, with_latent=True, with_dlogp=with_dlogp) # with_dlogp=False for now
        latent_np = np.append(latent_np, latent.detach().cpu().numpy())
        samples_np = np.append(samples_np, samples.detach().cpu().numpy())

        #dlogp_np = np.append(dlogp_np, as_numpy(dlogp))

    ### REALLY WANT TO ADD SOMETHING HERE WHICH SAVES WHAT THE SMALLEST LOSSES WERE FOR THAT BATCH... OR THEIR DISTRIBUTION???
    ### PERHAPS A GOOD WAY TO DO THAT WOULD BE TO PICKLE AND SAVE THE LOSS HANDLER

    latent_np = latent_np.reshape(-1, dim)
    samples_np = samples_np.reshape(-1, dim)

    np.savez(
        f"{save_dir}{save_data_name}_batch-{i}.npz",
        latent_np=latent_np,
        samples_np=samples_np,
        #dlogp_np=dlogp_np,
    )
    print(f'saved batch #{i}')


# Define the file path to save the loss_handler
loss_handler_save_path = f"{save_dir}{save_data_name}_loss_handler.pkl"

# Save the loss_handler using pickle
with open(loss_handler_save_path, "wb") as f:
    pickle.dump(loss_handler, f)

print(f"loss_handler saved to {loss_handler_save_path}")

print(f'''
      -------======= DONE SAMPLING WITH {filename} =======-------
      ''')