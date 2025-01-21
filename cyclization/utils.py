# batch-friendly loss building blocks
import torch
import mdtraj as md

def sq_distance(a1: torch.Tensor, a2: torch.Tensor) -> torch.Tensor:
    """
    Compute squared distances for batches of points.

    Parameters:
    a1, a2 (torch.Tensor): Tensors of shape (batch_size, 3)

    Returns:
    torch.Tensor: Squared distances of shape (batch_size, ).
    """

    result = torch.sum((a1 - a2) ** 2, dim=-1)
    
    return result

def bond_angle(a1: torch.Tensor, a2: torch.Tensor, a3: torch.Tensor) -> torch.Tensor:
    """
    Compute bond angles for batches of points.

    Computes the angle formed by vec(a2, a1) & vec(a2, a3)

    Args:
    ----
    a1, a2, a3 (torch.Tensor): Tensors of shape (batch_size, 3). 
    
    a2 is the "middle" atom that is the "crook" of the angle. a1 & a3 are the angle extremities.

    Returns:
    -------
    torch.Tensor: Bond angles of shape (batch_size, ), in radians.
    """
    
    v1 = a1 - a2 # position of a1 relative to a2
    v2 = a3 - a2 # position of a3 relative to a2

    # Normalize vectors
    v1_norm = v1 / torch.norm(v1, dim=-1, keepdim=True).clamp(min=1e-8)
    v2_norm = v2 / torch.norm(v2, dim=-1, keepdim=True).clamp(min=1e-8)

    # Compute cosine of angles
    cos_theta = torch.sum(v1_norm * v2_norm, dim=-1).clamp(-1.0, 1.0)
    result = torch.acos(cos_theta)

    return result

def dihedral_angle(a1: torch.Tensor, a2: torch.Tensor, # make sure to check the signs formed by the angles here...
                   a3: torch.Tensor, a4: torch.Tensor) -> torch.Tensor:
    """
    Compute dihedral angles for batches of points.

    Computes the dihedral angle where a1 is the first "horn," v23 forms the
    "joint" axis, and a4 is the second "horn."

    Parameters:
    a1, a2, a3, a4 (torch.Tensor): Tensors of shape (batch_size, 3).

    a2, a3 correspond to the line of intersection, whilst a1 & a4 correspond
    to the extreme points that form the tips of their respective triangles.

    Returns:
    torch.Tensor: Dihedral angles of shape (batch_size, ), in radians.
    """
    v1 = a2 - a1 # line from first horn to line of intersection (v12)
    v2 = a3 - a2 # line of intersection (v23)
    v3 = a4 - a3 # line from line of intersection to second horn (v34)

    # Normal vectors to planes
    n1 = torch.cross(v1, v2, dim=-1)
    n2 = torch.cross(v2, v3, dim=-1)

    # Normalize normal vectors
    n1_norm = n1 / torch.norm(n1, dim=-1, keepdim=True).clamp(min=1e-8)
    n2_norm = n2 / torch.norm(n2, dim=-1, keepdim=True).clamp(min=1e-8)

    # Dot product for cosine of dihedral angle
    cos_theta = torch.sum(n1_norm * n2_norm, dim=-1).clamp(-1.0, 1.0)

    # Compute the sign using a helper vector
    m1 = torch.cross(n1_norm, n2_norm, dim=-1)
    sign = torch.sign(torch.sum(m1 * v2, dim=-1))

    result = sign * torch.acos(cos_theta)

    return result

def distance_loss(a1: torch.Tensor, a2: torch.Tensor, 
                  target_distance: float, tolerance: float=0.0) -> torch.Tensor:
    """
    Compute distance losses with optional tolerance for batches.

    Parameters:
    a1, a2 (torch.Tensor): Tensors of shape [batch_size, 3].
    target_distance (float): Target bond distance.
    tolerance (float): No-penalty range around the target distance.

    Returns:
    torch.Tensor: Losses of shape [batch_size, ].
    """
    sq_dist = sq_distance(a1, a2)
    dist = torch.sqrt(sq_dist)
    error = torch.abs(dist - target_distance)
    
    # Apply tolerance: zero penalty within tolerance range
    penalty = torch.where(error <= tolerance, torch.zeros_like(error), (error - tolerance) ** 2)

    result = penalty

    return result

def bond_angle_loss(a1: torch.Tensor, a2: torch.Tensor, a3: torch.Tensor, 
                    target_angle: float, tolerance: float= 0.0) -> torch.Tensor:
    """
    Compute bond angle losses with tolerance for batches.

    Parameters:
    a1, a2, a3 (torch.Tensor): Tensors of shape (batch_size, 3).
    target_angle (float): Target bond angle in radians.
    tolerance (float): No-penalty range around the target angle.

    Returns:
    torch.Tensor: Losses of shape (batch_size).
    """
    b_angle = bond_angle(a1, a2, a3)
    error = torch.abs(b_angle - target_angle)
    
    # Apply tolerance
    penalty = torch.where(error <= tolerance, torch.zeros_like(error), (error - tolerance) ** 2)

    result = penalty

    return result

def dihedral_angle_loss(a1: torch.Tensor, a2: torch.Tensor, a3: torch.Tensor, a4: torch.Tensor, 
                        target_angle: float, tolerance: float= 0.0) -> torch.Tensor:
    """
    Compute dihedral angle losses with tolerance for batches.

    Computes a dihedral angle

    Parameters:
    a1, a2, a3, a4 (torch.Tensor): Tensors of shape (batch_size, 3).
    target_angle (float): Target dihedral angle in radians.
    tolerance (float): No-penalty range around the target angle in radians.

    Returns:
    torch.Tensor: Losses of shape (batch_size, ).
    """
    d_angle = dihedral_angle(a1, a2, a3, a4)
    error = torch.abs(d_angle - target_angle)
    
    # Apply tolerance
    penalty = torch.where(error <= tolerance, torch.zeros_like(error), (error - tolerance) ** 2)

    result = penalty

    return result # [batch_size, ]

def motif_absolute(*argv, target_structure): ### TODO: IMPLEMENT THIS
    """
    Compute a rotationally invariant loss based on relative structures.

    Parameters:
    positions (torch.Tensor): Tensor of shape (N, 3) representing current particle positions.
    reference_positions (torch.Tensor): Tensor of shape (N, 3) representing the reference structure.

    Returns:
    torch.Tensor: Scalar loss value.
    """
    
    pass

def soft_min(inputs: torch.Tensor, alpha=-3) -> torch.Tensor:
    """
    Compute soft minimum across batches; as alpha -> -inf, becomes a hard minimum. As alpha -> 0, becomes
    a simple average. as alpha -> +inf, becomes a hard maximum

    Parameters:
    ----------
    inputs (torch.Tensor): Tensor of shape (n_batch, n_losses). This represents each of the different cyclic
    losses for each batch

    alpha (float): Smoothness parameter.

    Returns:
    -------
    torch.Tensor: Soft minimum for each batch of shape (batch_size, ).
    """

    exps = torch.exp(alpha * inputs)
    result = torch.sum(inputs * exps, dim=-1) / torch.sum(exps, dim=-1)
    
    return result

# generates edges per the below scheme
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

# unsure of where else to put this
def inherit_docstring(parent_method):
    def decorator(method):
        method.__doc__ = parent_method.__doc__
        return method
    return decorator