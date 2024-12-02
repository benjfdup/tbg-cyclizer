import torch
import numpy as np
import mdtraj as md # addition by bfd

def sum_except_batch(x):
    return x.reshape(x.size(0), -1).sum(dim=-1)


def remove_mean(x):
    mean = torch.mean(x, dim=1, keepdim=True)
    x = x - mean
    return x


def remove_mean_with_mask(x, node_mask):
    assert (x * (1 - node_mask)).abs().sum().item() < 1e-8
    N = node_mask.sum(1, keepdims=True)

    mean = torch.sum(x, dim=1, keepdim=True) / N
    x = x - mean * node_mask
    return x


def assert_mean_zero(x):
    mean = torch.mean(x, dim=1, keepdim=True)
    assert mean.abs().max().item() < 1e-4


def assert_mean_zero_with_mask(x, node_mask):
    assert_correctly_masked(x, node_mask)
    assert torch.sum(x, dim=1, keepdim=True).abs().max().item() < 1e-4, \
        'Mean is not zero'


def assert_correctly_masked(variable, node_mask):
    assert (variable * (1 - node_mask)).abs().max().item() < 1e-4, \
        'Variables not masked properly.'


def center_gravity_zero_gaussian_log_likelihood(x):
    assert len(x.size()) == 3
    B, N, D = x.size()
    assert_mean_zero(x)

    # r is invariant to a basis change in the relevant hyperplane.
    r2 = sum_except_batch(x.pow(2))

    # The relevant hyperplane is (N-1) * D dimensional.
    degrees_of_freedom = (N-1) * D

    # Normalizing constant and logpx are computed:
    log_normalizing_constant = -0.5 * degrees_of_freedom * np.log(2*np.pi)
    log_px = -0.5 * r2 + log_normalizing_constant

    return log_px


def sample_center_gravity_zero_gaussian(size, device):
    assert len(size) == 3
    x = torch.randn(size, device=device)

    # This projection only works because Gaussian is rotation invariant around
    # zero and samples are independent!
    x_projected = remove_mean(x)
    return x_projected


def center_gravity_zero_gaussian_log_likelihood_with_mask(x, node_mask):
    assert len(x.size()) == 3
    B, N_embedded, D = x.size()
    assert_mean_zero_with_mask(x, node_mask)

    # r is invariant to a basis change in the relevant hyperplane, the masked
    # out values will have zero contribution.
    r2 = sum_except_batch(x.pow(2))

    # The relevant hyperplane is (N-1) * D dimensional.
    N = node_mask.squeeze(2).sum(1)  # N has shape [B]
    degrees_of_freedom = (N-1) * D

    # Normalizing constant and logpx are computed:
    log_normalizing_constant = -0.5 * degrees_of_freedom * np.log(2*np.pi)
    log_px = -0.5 * r2 + log_normalizing_constant

    return log_px


def sample_center_gravity_zero_gaussian_with_mask(size, device, node_mask):
    assert len(size) == 3
    x = torch.randn(size, device=device)

    x_masked = x * node_mask

    # This projection only works because Gaussian is rotation invariant around
    # zero and samples are independent!
    x_projected = remove_mean_with_mask(x_masked, node_mask)
    return x_projected


def standard_gaussian_log_likelihood(x):
    # Normalizing constant and logpx are computed:
    log_px = sum_except_batch(-0.5 * x * x - 0.5 * np.log(2*np.pi))
    return log_px


def sample_gaussian(size, device):
    x = torch.randn(size, device=device)
    return x


def standard_gaussian_log_likelihood_with_mask(x, node_mask):
    # Normalizing constant and logpx are computed:
    log_px_elementwise = -0.5 * x * x - 0.5 * np.log(2*np.pi)
    log_px = sum_except_batch(log_px_elementwise * node_mask)
    return log_px


def sample_gaussian_with_mask(size, device, node_mask):
    x = torch.randn(size, device=device)
    x_masked = x * node_mask
    return x_masked

def create_adjacency_list(distance_matrix, atom_types):
    adjacency_list = []

    # Iterate through the distance matrix
    num_nodes = len(distance_matrix)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):  # Avoid duplicate pairs
            distance = distance_matrix[i][j]
            element_i = atom_types[i]
            element_j = atom_types[j]
            if 1 in (element_i, element_j):
                distance_cutoff = 0.14
            elif 4 in (element_i, element_j):
                distance_cutoff = 0.22
            elif 0 in (element_i, element_j):
                distance_cutoff = 0.18
            else:
                # elements should not be bonded
                distance_cutoff = 0.0

            # Add edge if distance is below the cutoff
            if distance < distance_cutoff:
                adjacency_list.append([i,j])

    return adjacency_list

# chekc if chirality is the same
# if not --> mirror
# if still not --> discard
def find_chirality_centers(
    adj_list: torch.Tensor, atom_types: torch.Tensor, num_h_atoms: int = 2
) -> torch.Tensor:
    """
    Return the chirality centers for a peptide, e.g. carbon alpha atoms and their bonds.

    Args:
        adj_list: List of bonds
        atom_types: List of atom types
        num_h_atoms: If num_h_atoms or more hydrogen atoms connected to the center, it is not reportet.
            Default is 2, because in this case the mirroring is a simple permutation.

    Returns:
        chirality_centers
    """
    chirality_centers = []
    candidate_chirality_centers = torch.where(torch.unique(adj_list, return_counts=True)[1] == 4)[0]
    for center in candidate_chirality_centers:
        bond_idx, bond_pos = torch.where(adj_list == center)
        bonded_idxs = adj_list[bond_idx, (bond_pos + 1) % 2].long()
        adj_types = atom_types[bonded_idxs]
        if torch.count_nonzero(adj_types - 1) > num_h_atoms:
            chirality_centers.append([center, *bonded_idxs[:3]])
    return torch.tensor(chirality_centers).to(adj_list).long()


def compute_chirality_sign(coords: torch.Tensor, chirality_centers: torch.Tensor) -> torch.Tensor:
    """
    Compute indicator signs for a given configuration.
    If the signs for two configurations are different for the same center, the chirality changed.

    Args:
        coords: Tensor of atom coordinates
        chirality_centers: List of chirality_centers

    Returns:
        Indicator signs
    """
    assert coords.dim() == 3
    # print(coords.shape, chirality_centers.shape, chirality_centers)
    direction_vectors = (
        coords[:, chirality_centers[:, 1:], :] - coords[:, chirality_centers[:, [0]], :]
    )
    perm_sign = torch.einsum(
        "ijk, ijk->ij",
        direction_vectors[:, :, 0],
        torch.cross(direction_vectors[:, :, 1], direction_vectors[:, :, 2], dim=-1),
    )
    return torch.sign(perm_sign)


def check_symmetry_change(
    coords: torch.Tensor, chirality_centers: torch.Tensor, reference_signs: torch.Tensor
) -> torch.Tensor:
    """
    Check for a batch if the chirality changed wrt to some reference reference_signs.
    If the signs for two configurations are different for the same center, the chirality changed.

    Args:
        coords: Tensor of atom coordinates
        chirality_centers: List of chirality_centers
        reference_signs: List of reference sign for the chirality_centers
    Returns:
        Mask, where changes are True
    """
    perm_sign = compute_chirality_sign(coords, chirality_centers)
    return (perm_sign != reference_signs.to(coords)).any(dim=-1)

### Additions by bfd ###
def generate_bb_all_sc_adjacent_from_pdb(pdb_file: str):
    """
    Generates a custom adjacency matrix based on protein backbone and side-chain rules,
    using atom and residue information from an mdtraj topology.

    Backbone atoms are fully connected, while side-chain atoms are connected
    within their amino acid, to side-chains of adjacent amino acids, and to backbone
    atoms of their and adjacent amino acids.

    Args:
        pdb_file (str): Path to the .pdb file.

    Returns:
        torch.Tensor: Sparse adjacency matrix (shape [2, num_edges]).
    """

    # need to manually review the logic here...

    # Load structure using mdtraj
    traj = md.load(pdb_file)
    topology = traj.topology

    # Identify backbone and side-chain atoms
    backbone_atoms = [atom.index for atom in topology.atoms if atom.is_backbone]

    # Get residue indices
    amino_acid_indices = []
    for residue in topology.residues:
        start = residue.atom(0).index
        end = residue.atom(-1).index
        amino_acid_indices.append((start, end))

    edges = []

    # Backbone atoms are fully connected
    for i in backbone_atoms: # could likely speed this up... but probably doesn't matter.
        for j in backbone_atoms:
            if i != j:
                edges.append((i, j))

    # Side-chain atoms connect within the amino acid, to adjacent amino acids, and to backbone atoms
    for idx, (start, end) in enumerate(amino_acid_indices):
        # Separate backbone and side-chain atoms within this amino acid
        residue_atoms = list(range(start, end + 1))
        residue_backbone = [atom for atom in residue_atoms if atom in backbone_atoms]
        residue_sidechain = [atom for atom in residue_atoms if atom not in backbone_atoms]

        # Connect side-chain atoms within the amino acid
        for i in residue_sidechain:
            for j in residue_sidechain:
                if i != j:
                    edges.append((i, j))

        # Connect side-chain atoms to backbone atoms in the same amino acid
        for i in residue_sidechain:
            for j in residue_backbone:
                edges.append((i, j))
                edges.append((j, i))  # Ensure bidirectional connectivity

        # Connect side-chain atoms to side-chain atoms and backbone atoms in adjacent amino acids
        if idx > 0:  # Connect to the previous residue
            prev_start, prev_end = amino_acid_indices[idx - 1]
            prev_atoms = list(range(prev_start, prev_end + 1))
            prev_backbone = [atom for atom in prev_atoms if atom in backbone_atoms]
            prev_sidechain = [atom for atom in prev_atoms if atom not in backbone_atoms]

            for i in residue_sidechain:
                # Connect to side-chains of the previous residue
                for j in prev_sidechain:
                    edges.append((i, j))
                    edges.append((j, i))  # Bidirectional

                # Connect to backbone of the previous residue
                for j in prev_backbone:
                    edges.append((i, j))
                    edges.append((j, i))  # Bidirectional

        if idx < len(amino_acid_indices) - 1:  # Connect to the next residue
            next_start, next_end = amino_acid_indices[idx + 1]
            next_atoms = list(range(next_start, next_end + 1))
            next_backbone = [atom for atom in next_atoms if atom in backbone_atoms]
            next_sidechain = [atom for atom in next_atoms if atom not in backbone_atoms]

            for i in residue_sidechain:
                # Connect to side-chains of the next residue
                for j in next_sidechain:
                    edges.append((i, j))
                    edges.append((j, i))  # Bidirectional

                # Connect to backbone of the next residue
                for j in next_backbone:
                    edges.append((i, j))
                    edges.append((j, i))  # Bidirectional

    # Convert to tensor
    edges = torch.tensor(edges, dtype=torch.long).T  # Shape: [2, num_edges]
    #print('DONE generating adjacency matrix')
    return edges
