### IMPORTS & SETUP ###
import torch
import numpy as np

from bgflow.utils import (
    IndexBatchIterator,
)
from bgflow import (
    DiffEqFlow, # look into how this works
    MeanFreeNormalDistribution, # look into how this works, too.
)
from tbg.models2 import EGNN_dynamics_transferable_MD
from bgflow import BlackBoxDynamics, BruteForceEstimator
import os
import tqdm
import mdtraj as md
from torch.utils.tensorboard import SummaryWriter

### Directory and Peptide Information Setup ###
data_path = "/home/bfd21/rds/hpc-work/bfd_2AA-dummy"
n_dimensions = 3

### Training Peptides ###
directory = os.fsencode(data_path + "/train") # loading training peptides
training_peptides = [] # the peptides that will ultimately be used for training.

for file in os.listdir(directory): # loop just fills the training peptides
    filename = os.fsdecode(file)
    if filename.endswith(".pdb"): # only storing the ones that have valid data about them in the directory.
        training_peptides.append(filename[:2]) # appending the first 2 letters of the filename.

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

max_atom_number = 51 # i'd like to compute this rather than have it hard coded in as such. This is because when I move to 
# n-peptides this wont be true anymore.
# unsure as to why this is 51...
# can probably change this based on my dummy training set... but lets leave this for now and see what happens.

#max_atom_number = int(max(np.max(validation_n_atoms_list), np.max(training_n_atoms_list)))

atom_dict = {"H": 0, "C": 1, "N": 2, "O": 3, "S": 4}
scaling = 30 # scaling factor used to compute how distances are scaled from the CoM

priors = {} # stores the priors for each atom in each peptide from which initial points are drawn.
topologies = {} # stores the topologies of all the atoms in each peptide
atom_types_dict = {} # stores the type of atom in each peptide
h_dict = {} # holds everything about the peptide.
n_encodings = 76 # the length of the final h_dict encodings. May need to change this for future examples.
atom_types_ecoding = np.load(
    data_path + "/atom_types_ecoding.npy", allow_pickle=True
).item()

training_n_atoms_list = []

for peptide in tqdm.tqdm(training_peptides): # looping over all training peptides. Assembling their information.

    topologies[peptide] = md.load_topology(
        data_path + f"/train/{peptide}-traj-state0.pdb" # encodes the bond topology of the atoms encoded.
    )
    n_atoms = len(list(topologies[peptide].atoms)) # number of atoms in the given dipeptide
    training_n_atoms_list.append(n_atoms) # bfd added this to manually compute the max_atoms needed later

    atom_types = []
    amino_idx = []
    amino_types = []
    for i, amino in enumerate(topologies[peptide].residues): # looping over the individual amino acids in the dipeptide.

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
    
    atom_types_dict[peptide] = np.array( # encodes information on the atom type, as well as its place in the amino acid
        # encodes values 0 - 53, inclusive.
        [atom_types_ecoding[atom_type] for atom_type in atom_types]
    )

    atom_onehot = torch.nn.functional.one_hot( # converts each atom type to a one hot encoding.
        torch.tensor(atom_types_dict[peptide]), num_classes=len(atom_types_ecoding) # 54 classes. changes with ecoding.
    ) # makes a tensor of shape (n, 54), where n is the number of atoms in the dipeptide, and 54 is the number of atom types.

    amino_idx_onehot = torch.nn.functional.one_hot( # encodes the position as a 1-hot vector. Note that num_classes = 2 because
        # we are only working with di-peptides here...
        torch.tensor(amino_idx), num_classes=2
    )

    amino_types_onehot = torch.nn.functional.one_hot( # one hot encoding of the amino acid type. 20 different kinds of amino acid,
        # hence, 20 classes.
        # of shape (n, 20), where n is the number of atoms and 20 is the number of amino acid classes here...
        torch.tensor(amino_types), num_classes=20
    )

    h_dict[peptide] = torch.cat(
        [amino_idx_onehot, amino_types_onehot, atom_onehot], dim=1 # concatinates all of this information along the first axis
        # ie: along each atom. Therefore h_dict[peptide] is ultimately of shape: (n, 54 + 2 + 20)
        # or, put another way, h_dict[peptide] is of shape (n, len(atom_types_ecoding) + aa_length + #_of_amino_acids)
    )

    priors[peptide] = MeanFreeNormalDistribution( # I'd like to poke around in here to see where the std is defined.
        # defines the prior distributions from which the positions of each atom are sampled;
        # we apply the learned vector field to this to move the atoms into the data-like distribution.
        n_atoms * n_dimensions, n_atoms, two_event_dims=False
    ).cuda()

### Validation Peptides ###

validation_n_atoms_list = []

directory = os.fsencode(data_path + "/val")
validation_peptides = []
val_priors = {}
val_topologies = {}
val_atom_types_dict = {}
val_h_dict = {}
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".pdb"):
        validation_peptides.append(filename[:2])

# Looping over the validation peptides
for peptide in tqdm.tqdm(validation_peptides):

    val_topologies[peptide] = md.load_topology(
        data_path + f"/val/{peptide}-traj-state0.pdb"
    )

    n_atoms = len(list(val_topologies[peptide].atoms))
    validation_n_atoms_list.append(n_atoms)

    atom_types = []
    amino_idx = []
    amino_types = []
    for i, amino in enumerate(val_topologies[peptide].residues):

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
    val_atom_types_dict[peptide] = np.array(
        [atom_types_ecoding[atom_type] for atom_type in atom_types]
    )
    atom_onehot = torch.nn.functional.one_hot(
        torch.tensor(val_atom_types_dict[peptide]), num_classes=len(atom_types_ecoding)
    )
    amino_idx_onehot = torch.nn.functional.one_hot(
        torch.tensor(amino_idx), num_classes=2
    )
    amino_types_onehot = torch.nn.functional.one_hot(
        torch.tensor(amino_types), num_classes=20
    )

    val_h_dict[peptide] = torch.cat(
        [amino_idx_onehot, amino_types_onehot, atom_onehot], dim=1
    )
    val_priors[peptide] = MeanFreeNormalDistribution(
        n_atoms * n_dimensions, n_atoms, two_event_dims=False
    ).cuda()
# notes are the same for the validation peptides as for the training.

### Loading training and validation data.
data = np.load(data_path + "/bfd_dummy_all_train.npy", allow_pickle=True).item() # dictionary containing final "true" conformations... ?
data_val = np.load(data_path + "/bfd_dummy_all_val.npy", allow_pickle=True).item() # need to find this... where is this generated.

n_data = len(data[training_peptides[0]]) # number of datapoints (per peptide). 9800 for my dummy dataset.
n_random = n_data // 10 # divided by 10 and rounded to the nearest integer. 980 for my dummy dataset.

n_layers = 9 # How many times our algorithm iterates at sample time.
hidden_nf = 128 # for speed's sake, all NNs only have one layer, are wide. this is the number of hidden features per layer.
net_dynamics = EGNN_dynamics_transferable_MD( # look into the code of this... step into this. Defines the Phi's used in the system
    n_particles=max_atom_number, # the maximum number of nodes to consider (some of which will be masked out)
    h_size=n_encodings, # the length of each h_dict encoding
    device="cuda", # the device its all to be run on. Should be relatively straight forward.
    n_dimension=n_dimensions, # 3
    hidden_nf=hidden_nf, # each network only has one hidden layer for speed purposes.
    act_fn=torch.nn.SiLU(),
    n_layers=n_layers, # the amount of iterations between each input and the final vector field

    ### REVIEW WHAT THESE ARGUMENTS DO ###
    recurrent=True, # unsure... Probably just that the network takes its old outputs back as another input.
    tanh=True, # Really unsure on this one...
    attention=True, # Really unsure on this one. Probably just that it uses some attention mechanism?
    condition_time=True, # No idea...
    ### REVIEW WHAT THESE ARGUMENTS DO ###

    mode="egnn_dynamics", # probably just that the output will be a vector that updates the position.
    agg="sum", # how messages are aggregated; principle requirement is that this operator is invariant of the order of reciept
    # maybe to avoid creating race conditions?
)

bb_dynamics = BlackBoxDynamics( # need to review this, but this seems to be what defines the dynamics (ODE) which are fed into the solve
    dynamics_function=net_dynamics, divergence_estimator=BruteForceEstimator()
)

flow = DiffEqFlow(dynamics=bb_dynamics) # this seems to be the actual ODE solver.

n_batch = 2 # batch size for for training.
n_batch_val = 20 # batch size of the validation set.

def resample_noise( # samples the initial positions from the 
    peptides=training_peptides,
    priors=priors,
    h_dict=h_dict,
    n_samples=n_random, # look up exactly what this is doing
    n_batch=n_batch,
):
    # initializing outputs
    x0_list = []
    noise_list = []

    for peptide in peptides: # loops over the relevant training peptides.
        n_particles = h_dict[peptide].shape[0] # getting the number of particles within the given peptide

        x0 = ( # generates the initial coordinates of the sample to be used in training.
            priors[peptide]
            .sample(n_batch * n_samples) # how many samples to draw per each batch.
            .cpu() # to be done with cpu
            .reshape(n_batch, n_samples, -1) # to be flattened along its last dimension (so that its more ameniable to nn analysis)
        )
        x0 = torch.nn.functional.pad(x0, (0, (max_atom_number - n_particles) * 3)) # pads remaining particles with zeros.

        noise = ( # generates noise in the same way that it generates each particle
            priors[peptide] # drawn from the same prior. Unsure of why are generating both noise and a starting position?
            .sample(n_batch * n_samples)
            .cpu()
            .reshape(n_batch, n_samples, -1)
        )
        noise = torch.nn.functional.pad(noise, (0, (max_atom_number - n_particles) * 3)) # pads the noise similarly.

        x0_list.append(x0) # adds the samples to a list for storage.
        noise_list.append(noise) # adds the noise to a list for storage.

    x0_list = torch.cat(x0_list) # concatenates the list into a tensor for ease of use
    noise_list = torch.cat(noise_list) # ""
    return x0_list, noise_list

val_x1_list = [] # will store the "ground truth" conformational positions of the validation peptides
val_node_mask_batch = [] # stores the node mask for the validation peptide conformations (which atoms to mask)
val_edge_mask_batch = [] # stores the edge mask for the validation peptide conformations (which interactions to ignore)
val_h_batch = [] # stores the encoding vectors for each point in the relevant peptides

for peptide in validation_peptides:
    n_particles = val_h_dict[peptide].shape[0] # getting the number of particles within the given peptide

    x1 = torch.from_numpy(data_val[peptide]) # getting the ground truth positions of the given peptide.
    x1 = torch.nn.functional.pad(x1, (0, (max_atom_number - n_particles) * 3)) # padding the peptide nodes.
    val_x1_list.append(x1) # appends it to list.
    # create the masks here as well!

    mask = torch.ones((n_batch_val, n_particles)) # creates node masks.
    # node mask as bool ornot???

    mask = torch.nn.functional.pad( # creates node masks.
        mask, (0, (max_atom_number - n_particles)) # masks everything that outside of the number of particles for that given peptide.
    )  # .bool()

    edge_mask = mask.unsqueeze(1) * mask.unsqueeze(2) # go over more thoroughly, though I think I understand this.
    # mask diagonal

    ### START HERE, WHEN GOING THROUGH NEXT TIME, BEN. ###

    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0) # putting 1s at the diagonals?
    edge_mask *= diag_mask
    node_mask = mask
    edge_mask = edge_mask.reshape(-1, max_atom_number**2) # creating the relevant node and edge masks.
    h = ( # sort of unsure of the point of this. Is this creating the h's for the validation peptides? Seems like it?
        torch.cat(
            [
                val_h_dict[peptide],
                torch.zeros(max_atom_number - n_particles, n_encodings),
            ]
        )
        .unsqueeze(0)
        .repeat(n_batch_val, 1, 1)
    )
    val_node_mask_batch.append(node_mask)
    val_edge_mask_batch.append(edge_mask)
    val_h_batch.append(h)
    
val_x1_list = torch.stack(val_x1_list) # creates a list of the ground truth validation positions
val_node_mask_batch = torch.cat(val_node_mask_batch, dim=0).cuda() # concatenates the validation masks
val_edge_mask_batch = torch.cat(val_edge_mask_batch, dim=0).cuda() # concatenates the edges for the batches?
val_h_batch = torch.cat(val_h_batch, dim=0).cuda() # concatenates the h's for the validation batches?

### THIS SEEMS TO BE FOR THE TRAINING PEPTIDES ###
x1_list = [] # creates the ground truth list of final positions
node_mask_batch = [] # creates the node mask batches?
edge_mask_batch = [] # same for edges?
h_batch = [] # creates the h's for the batches?
### YUP, FOR TRAINING PEPTIDES ###

for peptide in training_peptides:
    n_particles = h_dict[peptide].shape[0] # Number of atoms in the current peptide

    x1 = torch.from_numpy(data[peptide])
    x1 = torch.nn.functional.pad(x1, (0, (max_atom_number - n_particles) * 3))
    x1_list.append(x1)
    # create the masks here as well!
    mask = torch.ones((n_batch, n_particles))
    # node mask as bool ornot???
    mask = torch.nn.functional.pad(
        mask, (0, (max_atom_number - n_particles))
    )  # .bool()
    edge_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
    # mask diagonal
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask
    node_mask = mask
    edge_mask = edge_mask.reshape(-1, max_atom_number**2)
    h = (
        torch.cat(
            [h_dict[peptide], torch.zeros(max_atom_number - n_particles, n_encodings)]
        )
        .unsqueeze(0)
        .repeat(n_batch, 1, 1)
    )
    node_mask_batch.append(node_mask)
    edge_mask_batch.append(edge_mask)
    h_batch.append(h)
x1_list = torch.stack(x1_list)
node_mask_batch = torch.cat(node_mask_batch, dim=0).cuda()
edge_mask_batch = torch.cat(edge_mask_batch, dim=0).cuda()
h_batch = torch.cat(h_batch, dim=0).cuda()

### ^^^ LOOP SEEMS TO DO THE SAME THING BUT FOR TRAINING DATA ###

batch_iter = IndexBatchIterator(n_data, n_batch)
val_batch_iter = IndexBatchIterator(n_data, n_batch_val)

optim = torch.optim.Adam(flow.parameters(), lr=5e-4)

n_epochs = 12 # the number of epochs of training data

### TRAINING STARTS HERE ###

PATH_last = f"/home/bfd21/rds/hpc-work/tbg/bfd_models/tbg_full"
writer = SummaryWriter("/home/bfd21/rds/hpc-work/logs/" + 'tbg_full')
try: # tries to load the extant model from a checkpoint
    checkpoint = torch.load(PATH_last)
    flow.load_state_dict(checkpoint["model_state_dict"])
    optim.load_state_dict(checkpoint["optimizer_state_dict"])
    loaded_epoch = checkpoint["epoch"]
    global_it = checkpoint["global_it"]
    print(f"Successfully loaded model {PATH_last}")
except: # generates a new model
    print("Generated new model")
    loaded_epoch = 0
    global_it = 0


sigma = 0.01 # the std of the samples and data throughout time.
for epoch in range(loaded_epoch, n_epochs):
    if epoch == 4:
        for g in optim.param_groups:
            g["lr"] = 5e-5
    if epoch == 8:
        for g in optim.param_groups:
            g["lr"] = 5e-6
    random_start_idx = torch.randint(0, n_data, (len(x1_list),)).unsqueeze(1)
    for it, idxs in enumerate(batch_iter):
        if len(idxs) != n_batch:
            continue
        peptide_idxs = torch.arange(0, len(x1_list)).repeat_interleave(len(idxs))
        it_idxs = it % n_random
        if it_idxs == 0:
            x0_list, noise_list = resample_noise()
            print(epoch, it)
            torch.save(
                {
                    "model_state_dict": flow.state_dict(),
                    "optimizer_state_dict": optim.state_dict(),
                    "epoch": epoch,
                    "global_it": global_it,
                },
                PATH_last,
            )
        optim.zero_grad()
        x1 = x1_list[
            peptide_idxs, ((random_start_idx + idxs) % n_data).flatten()
        ].cuda()

        batchsize = x1.shape[0]
        t = torch.rand(len(x1), 1).to(x1)

        x0 = x0_list[:, it_idxs].to(x1)
        noise = noise_list[:, it_idxs].to(x1)

        mu_t = x0 * (1 - t) + x1 * t
        sigma_t = sigma
        x = mu_t + sigma_t * noise
        ut = x1 - x0
        vt = flow._dynamics._dynamics._dynamics_function(
            t, x, h_batch, node_mask_batch, edge_mask_batch
        )
        # loss = torch.mean((vt - ut_batch) ** 2)
        # use the weighted loss instead
        loss = (
            torch.sum((vt - ut) ** 2, dim=-1)
            / node_mask_batch.int().sum(-1)
            / n_dimensions
        )
        loss = loss.mean()
        loss.backward()
        optim.step()
        writer.add_scalar("Loss/Train", loss, global_step=global_it)
        global_it += 1
    print("Validating")
    with torch.no_grad():
        loss_acum = 0
        random_start_idx = torch.randint(0, n_data, (len(val_x1_list),)).unsqueeze(1)
        for it, idxs in enumerate(val_batch_iter):
            if it == 100:
                break
            peptide_idxs = torch.arange(0, len(val_x1_list)).repeat_interleave(
                len(idxs)
            )
            x0_list, noise_list = resample_noise(
                validation_peptides, val_priors, val_h_dict, n_batch=n_batch_val
            )
            # print(val_x1_list.shape, peptide_idxs.shape, idxs.shape)
            x1 = val_x1_list[
                peptide_idxs, ((random_start_idx + idxs) % n_data).flatten()
            ].cuda()
            batchsize = x1.shape[0]
            t = torch.rand(len(x1), 1).to(x1)

            x0 = x0_list[:, it].to(x1)
            noise = noise_list[:, it].to(x1)
            # print(x1.shape, x0.shape)

            mu_t = x0 * (1 - t) + x1 * t
            sigma_t = sigma
            x = mu_t + sigma_t * noise
            ut = x1 - x0
            vt = flow._dynamics._dynamics._dynamics_function(
                t, x, val_h_batch, val_node_mask_batch, val_edge_mask_batch
            )
            # loss = torch.mean((vt - ut_batch) ** 2)
            # use the weighted loss instead
            loss = (
                torch.sum((vt - ut) ** 2, dim=-1)
                / val_node_mask_batch.int().sum(-1)
                / n_dimensions
            )
            loss_acum += loss.mean()
    writer.add_scalar("Loss/Val", loss_acum / 100, global_step=global_it)