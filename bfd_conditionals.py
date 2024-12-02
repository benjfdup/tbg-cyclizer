### Imports ###

import torch
import numpy as np

import os
import tqdm
import mdtraj as md
import sys

### INSERT PHYSICAL QUANTITIES HERE, AS WELL AS RELEVANT SCALE FACTORS ###
scaling_factor = 30 # the factor of scaling in question. Will need to account for this, l8r
## ^^^ ###

def sq_distance(a1, a2):
    a1 = torch.tensor(a1, requires_grad=True, device="cuda" if torch.cuda.is_available() else "cpu")
    a2 = torch.tensor(a2, requires_grad=True, device="cuda" if torch.cuda.is_available() else "cpu")

    # Calculate squared distance
    return torch.sum((a1 - a2) ** 2)

def bond_angle(a1, a2, a3):
    # Convert inputs to PyTorch tensors if they are not already
    a1 = torch.tensor(a1, requires_grad=True, device="cuda" if torch.cuda.is_available() else "cpu")
    a2 = torch.tensor(a2, requires_grad=True, device="cuda" if torch.cuda.is_available() else "cpu")
    a3 = torch.tensor(a3, requires_grad=True, device="cuda" if torch.cuda.is_available() else "cpu")
    
    v1 = a1 - a2
    v2 = a3 - a2

    # Normalize vectors to get unit vectors
    v1_norm = v1 / torch.norm(v1)
    v2_norm = v2 / torch.norm(v2)

    # Compute the cosine of the angle using dot product
    cos_theta = torch.dot(v1_norm, v2_norm).clamp(-1.0, 1.0)  # Clamp to avoid numerical errors out of range

    # Compute the angle in radians
    angle = torch.acos(cos_theta)

    return angle

def dihedral_angle(a1, a2, a3, a4): ### GO OVER THE INNARDS OF THIS FUNCTION! ###
    # Convert inputs to PyTorch tensors if they are not already
    if not isinstance(a1, torch.Tensor):
        a1 = torch.tensor(a1, requires_grad=True, device="cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(a2, torch.Tensor):
        a2 = torch.tensor(a2, requires_grad=True, device="cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(a3, torch.Tensor):
        a3 = torch.tensor(a3, requires_grad=True, device="cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(a4, torch.Tensor):
        a4 = torch.tensor(a4, requires_grad=True, device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Calculate vectors
    v1 = a2 - a1
    v2 = a3 - a2
    v3 = a4 - a3

    # Calculate normal vectors of the planes formed by (a1, a2, a3) and (a2, a3, a4)
    n1 = torch.cross(v1, v2)
    n2 = torch.cross(v2, v3)

    # Normalize the normal vectors
    n1_norm = n1 / torch.norm(n1)
    n2_norm = n2 / torch.norm(n2)

    # Calculate the angle between the planes (dihedral angle) using the dot product
    cos_theta = torch.dot(n1_norm, n2_norm).clamp(-1.0, 1.0)  # Clamp to avoid numerical issues

    # Calculate the sign of the dihedral angle using the direction of v2
    m1 = torch.cross(n1, n2)
    sign = torch.sign(torch.dot(m1, v2))

    # Compute the angle in radians and apply the sign
    angle = sign * torch.acos(cos_theta)

    return angle

### ACTUAL LOSS FUNCTIONS DOWN HERE: ###
### NEED TO IMPLEMENT SOME STEEPNESS PARAMETER TO PLAY AROUND WITH ###

def cyclic_distance_loss(a1, a2, target_distance: float): ### I'd like to add some parameter here to either
    # 1.) not really penalize distances with some range of what is plausible or 2.) to encode a parabolic
    # steepness parameter which encodes the distances over which the "corrective" derivatives arent that 
    # steep (ie: the range of errors that we don't really care about)

    dist = torch.sqrt(sq_distance(a1, a2))

    return (dist - target_distance) ** 2

def bond_angle_loss(a1, a2, a3, target_angle: float):
    b_angle = bond_angle(a1, a2, a3)

    return (b_angle - target_angle) ** 2

def dihedral_angle_loss(a1, a2, a3, a4, target_angle: float):
    d_angle = dihedral_angle(a1, a2, a3, a4)

    return (d_angle - target_angle) ** 2

def soft_min(*argv, alpha = -10): # this will take losses as its input (in the argv part)
    # Stack the input tensors for efficient operations

    # all arvs must be greater than or equal to zero.

    # Seems to be working properly. Only gradient descent will tell...

    inputs = torch.stack(argv)

    # Ensure all inputs are on the same device (move to CUDA if needed)
    if inputs.device != torch.device('cuda'):
        inputs = inputs.to('cuda')

    exps = torch.exp(alpha * inputs)

    soft_min_result = torch.sum(inputs * exps) / torch.sum(exps)

    return soft_min_result

def motif_absolute(*argv, target_structure): # need to actually fill in this function...
    """
    Compute a rotationally invariant loss based on relative structures.

    Parameters:
    positions (torch.Tensor): Tensor of shape (N, 3) representing current particle positions.
    reference_positions (torch.Tensor): Tensor of shape (N, 3) representing the reference structure.

    Returns:
    torch.Tensor: Scalar loss value.
    """
    
    pass

### Specific Losses:

def disulfide_loss(cys1_s, cys1_cb, cys2_s, cys2_cb, 
                   target_distance=2.05, 
                   target_bond_angle=np.radians(110),  # ~110 degrees typical
                   target_dihedral_angle=np.radians(90),  # ~90 degrees typical
                   steepness=1,  # Controls how sharply deviations are penalized
                   distance_tolerance=0.2):  # Range of acceptable distances without penalty
    """
    Compute the disulfide loss for two cysteine residues.
    
    Parameters:
    cys1_s (torch.Tensor): Sulfur atom position of the first cysteine.
    cys1_cb (torch.Tensor): Beta carbon position of the first cysteine.
    cys2_s (torch.Tensor): Sulfur atom position of the second cysteine.
    cys2_cb (torch.Tensor): Beta carbon position of the second cysteine.
    target_distance (float): Ideal S-S bond distance in Å.
    target_bond_angle (float): Ideal bond angle in radians.
    target_dihedral_angle (float): Ideal dihedral angle in radians.
    steepness (float): Controls steepness of penalties.
    distance_tolerance (float): Acceptable deviation in distance without penalty.

    Returns:
    torch.Tensor: Loss value encoding deviations from the target geometry.
    """

    # Distance Loss
    dist = torch.sqrt(sq_distance(cys1_s, cys2_s))
    if abs(dist - target_distance) <= distance_tolerance:
        distance_loss = torch.tensor(0.0, device=dist.device)  # No penalty within tolerance
    else:
        distance_loss = steepness * (dist - target_distance) ** 2

    # Bond Angle Losses (Cβ-S-S and S-S-Cβ)
    angle1_loss = bond_angle_loss(cys1_cb, cys1_s, cys2_s, target_bond_angle)
    angle2_loss = bond_angle_loss(cys1_s, cys2_s, cys2_cb, target_bond_angle)

    # Dihedral Angle Loss (Cβ-S-S-Cβ)
    dihedral_loss = dihedral_angle_loss(cys1_cb, cys1_s, cys2_s, cys2_cb, target_dihedral_angle)

    # Combine losses
    #total_loss = soft_min(distance_loss, angle1_loss, angle2_loss, dihedral_loss, alpha=-steepness)
    total_loss = distance_loss + angle1_loss + angle2_loss + dihedral_loss

    return total_loss

def amide_loss(c1, ca1, n2, h2, 
               target_distance=1.33,  # Typical C-N bond length in Å
               target_bond_angle=np.radians(120),  # Typical bond angles in radians
               target_dihedral_angle=0.0,  # Planarity implies a dihedral angle near 0 or 180 degrees
               steepness=1,  # Controls steepness of penalties
               distance_tolerance=0.1):  # Range of acceptable distances without penalty
    """
    Compute the loss for forming an amide bond between two residues.
    
    Parameters:
    c1 (torch.Tensor): Carbonyl carbon atom of the first residue.
    ca1 (torch.Tensor): Alpha carbon atom of the first residue.
    n2 (torch.Tensor): Amide nitrogen atom of the second residue.
    h2 (torch.Tensor): Hydrogen attached to the amide nitrogen of the second residue.
    target_distance (float): Ideal C-N bond distance in Å.
    target_bond_angle (float): Ideal bond angle in radians.
    target_dihedral_angle (float): Ideal dihedral angle in radians.
    steepness (float): Controls steepness of penalties.
    distance_tolerance (float): Acceptable deviation in distance without penalty.

    Returns:
    torch.Tensor: Loss value encoding deviations from the target geometry.
    """

    # Distance Loss (C-N bond)
    dist = torch.sqrt(sq_distance(c1, n2))
    if abs(dist - target_distance) <= distance_tolerance:
        distance_loss = torch.tensor(0.0, device=dist.device)  # No penalty within tolerance
    else:
        distance_loss = steepness * (dist - target_distance) ** 2

    # Bond Angle Losses
    angle1_loss = bond_angle_loss(ca1, c1, n2, target_bond_angle)  # Cα-C-N angle
    angle2_loss = bond_angle_loss(c1, n2, h2, target_bond_angle)  # C-N-H angle

    # Dihedral Angle Loss (Cα-C-N-H planarity)
    dihedral_loss = dihedral_angle_loss(ca1, c1, n2, h2, target_dihedral_angle)

    # Combine losses
    total_loss = distance_loss + angle1_loss + angle2_loss + dihedral_loss

    return total_loss

def side_chain_amide_loss(n_side_chain, c_carboxyl, side_chain_anchor, carboxyl_anchor,
                          target_distance=1.33,  # Typical amide bond distance in Å
                          target_bond_angle=np.radians(120),  # Typical amide bond angle
                          target_dihedral_angle=np.radians(0),  # Planarity of amide bond
                          steepness=1,
                          distance_tolerance=0.1):
    """
    Computes the loss for forming a side-chain amide bond between a side-chain nitrogen 
    (e.g., lysine or ornithine) and a carboxyl group (e.g., aspartic acid or glutamic acid).
    
    Parameters:
    n_side_chain (torch.Tensor): Nitrogen atom of the side chain.
    c_carboxyl (torch.Tensor): Carbon atom of the carboxyl group.
    side_chain_anchor (torch.Tensor): Anchor atom connected to the side-chain nitrogen.
    carboxyl_anchor (torch.Tensor): Anchor atom connected to the carboxyl carbon.
    target_distance (float): Ideal bond distance between nitrogen and carbon in Å.
    target_bond_angle (float): Ideal bond angle around the amide bond in radians.
    target_dihedral_angle (float): Ideal dihedral angle for planarity in radians.
    steepness (float): Factor controlling the penalty steepness for deviations.
    distance_tolerance (float): Range of acceptable distances without penalty.

    Returns:
    torch.Tensor: Combined loss for the distance, bond angles, and dihedral angle.
    """

    # Distance Loss
    dist = torch.sqrt(sq_distance(n_side_chain, c_carboxyl))
    if abs(dist - target_distance) <= distance_tolerance:
        distance_loss = torch.tensor(0.0, device=dist.device)
    else:
        distance_loss = steepness * (dist - target_distance) ** 2

    # Bond Angle Losses
    angle1_loss = bond_angle_loss(side_chain_anchor, n_side_chain, c_carboxyl, target_bond_angle)
    angle2_loss = bond_angle_loss(n_side_chain, c_carboxyl, carboxyl_anchor, target_bond_angle)

    # Dihedral Angle Loss (Planarity)
    dihedral_loss = dihedral_angle_loss(side_chain_anchor, n_side_chain, c_carboxyl, carboxyl_anchor, target_dihedral_angle)

    # Combine losses
    total_loss = distance_loss + angle1_loss + angle2_loss + dihedral_loss
    return total_loss

def thioether_loss(sulfur_atom, carbon_atom, sulfur_anchor, carbon_anchor,
                   target_distance=1.8,  # Typical S-C bond distance in Å
                   target_bond_angle=np.radians(109),  # Bond angle for sp3 hybridized atoms
                   target_dihedral_angle=np.radians(90),  # Typical dihedral angle
                   steepness=1,
                   distance_tolerance=0.2):
    """
    Computes the loss for forming a thioether bond between a sulfur atom (e.g., cysteine or methionine)
    and a carbon atom (e.g., from another residue's side chain).
    
    Parameters:
    sulfur_atom (torch.Tensor): Sulfur atom position.
    carbon_atom (torch.Tensor): Carbon atom position.
    sulfur_anchor (torch.Tensor): Anchor atom connected to the sulfur.
    carbon_anchor (torch.Tensor): Anchor atom connected to the carbon.
    target_distance (float): Ideal bond distance between sulfur and carbon in Å.
    target_bond_angle (float): Ideal bond angle around the thioether bond in radians.
    target_dihedral_angle (float): Ideal dihedral angle in radians.
    steepness (float): Factor controlling the penalty steepness for deviations.
    distance_tolerance (float): Range of acceptable distances without penalty.

    Returns:
    torch.Tensor: Combined loss for the distance, bond angles, and dihedral angle.
    """

    # Distance Loss
    dist = torch.sqrt(sq_distance(sulfur_atom, carbon_atom))
    if abs(dist - target_distance) <= distance_tolerance:
        distance_loss = torch.tensor(0.0, device=dist.device)
    else:
        distance_loss = steepness * (dist - target_distance) ** 2

    # Bond Angle Losses
    angle1_loss = bond_angle_loss(sulfur_anchor, sulfur_atom, carbon_atom, target_bond_angle)
    angle2_loss = bond_angle_loss(sulfur_atom, carbon_atom, carbon_anchor, target_bond_angle)

    # Dihedral Angle Loss (Planarity)
    dihedral_loss = dihedral_angle_loss(sulfur_anchor, sulfur_atom, carbon_atom, carbon_anchor, target_dihedral_angle)

    # Combine losses
    total_loss = distance_loss + angle1_loss + angle2_loss + dihedral_loss + steepness

    return total_loss

def ester_loss(oxygen_hydroxyl, carbon_carboxyl, hydroxyl_anchor, carboxyl_anchor,
               target_distance=1.4,  # Typical O-C bond distance in Å
               target_bond_angle=np.radians(120),  # Ester bond angles
               target_dihedral_angle=np.radians(0),  # Planarity of ester bond
               steepness=1,
               distance_tolerance=0.1):
    
    """
    Computes the loss for forming an ester bond between a hydroxyl group (e.g., serine or threonine) 
    and a carboxyl group (e.g., aspartic acid or glutamic acid).
    
    Parameters:
    oxygen_hydroxyl (torch.Tensor): Oxygen atom of the hydroxyl group.
    carbon_carboxyl (torch.Tensor): Carbon atom of the carboxyl group.
    hydroxyl_anchor (torch.Tensor): Anchor atom connected to the hydroxyl oxygen.
    carboxyl_anchor (torch.Tensor): Anchor atom connected to the carboxyl carbon.
    target_distance (float): Ideal bond distance between oxygen and carbon in Å.
    target_bond_angle (float): Ideal bond angle around the ester bond in radians.
    target_dihedral_angle (float): Ideal dihedral angle in radians.
    steepness (float): Factor controlling the penalty steepness for deviations.
    distance_tolerance (float): Range of acceptable distances without penalty.

    Returns:
    torch.Tensor: Combined loss for the distance, bond angles, and dihedral angle.
    """

    # Distance Loss
    dist = torch.sqrt(sq_distance(oxygen_hydroxyl, carbon_carboxyl))
    if abs(dist - target_distance) <= distance_tolerance:
        distance_loss = torch.tensor(0.0, device=dist.device)
    else:
        distance_loss = steepness * (dist - target_distance) ** 2

    # Bond Angle Losses
    angle1_loss = bond_angle_loss(hydroxyl_anchor, oxygen_hydroxyl, carbon_carboxyl, target_bond_angle)
    angle2_loss = bond_angle_loss(oxygen_hydroxyl, carbon_carboxyl, carboxyl_anchor, target_bond_angle)

    # Dihedral Angle Loss (Planarity)
    dihedral_loss = dihedral_angle_loss(hydroxyl_anchor, oxygen_hydroxyl, carbon_carboxyl, carboxyl_anchor, target_dihedral_angle)

    # Combine losses
    total_loss = distance_loss + angle1_loss + angle2_loss + dihedral_loss

    return total_loss

def hydrazone_loss(nitrogen_hydrazine, carbon_carbonyl, hydrazine_anchor, carbonyl_anchor,
                   target_distance=1.45,  # Typical N=C bond distance in Å
                   target_bond_angle=np.radians(120),  # Typical hydrazone bond angles
                   target_dihedral_angle=np.radians(0),  # Planarity of hydrazone bond
                   steepness=1,
                   distance_tolerance=0.1):
    """
    Computes the loss for forming a hydrazone bond between a hydrazine derivative and a carbonyl group,
    commonly used in chemical crosslinking.
    
    Parameters:
    nitrogen_hydrazine (torch.Tensor): Nitrogen atom of the hydrazine group.
    carbon_carbonyl (torch.Tensor): Carbon atom of the carbonyl group.
    hydrazine_anchor (torch.Tensor): Anchor atom connected to the hydrazine nitrogen.
    carbonyl_anchor (torch.Tensor): Anchor atom connected to the carbonyl carbon.
    target_distance (float): Ideal bond distance between nitrogen and carbon in Å.
    target_bond_angle (float): Ideal bond angle around the hydrazone bond in radians.
    target_dihedral_angle (float): Ideal dihedral angle in radians.
    steepness (float): Factor controlling the penalty steepness for deviations.
    distance_tolerance (float): Range of acceptable distances without penalty.

    Returns:
    torch.Tensor: Combined loss for the distance, bond angles, and dihedral angle.
    """

    # Distance Loss
    dist = torch.sqrt(sq_distance(nitrogen_hydrazine, carbon_carbonyl))
    if abs(dist - target_distance) <= distance_tolerance:
        distance_loss = torch.tensor(0.0, device=dist.device)
    else:
        distance_loss = steepness * (dist - target_distance) ** 2

    # Bond Angle Losses
    angle1_loss = bond_angle_loss(hydrazine_anchor, nitrogen_hydrazine, carbon_carbonyl, target_bond_angle)
    angle2_loss = bond_angle_loss(nitrogen_hydrazine, carbon_carbonyl, carbonyl_anchor, target_bond_angle)

    # Dihedral Angle Loss (Planarity)
    dihedral_loss = dihedral_angle_loss(hydrazine_anchor, nitrogen_hydrazine, carbon_carbonyl, carbonyl_anchor, target_dihedral_angle)

    # Combine losses
    total_loss = distance_loss + angle1_loss + angle2_loss + dihedral_loss

    return total_loss

### NOW GENERATE THE TOTAL LOSS OF CYCLIZATION FROM A PDB FILE

def calculate_cyclization_losses(pdb_path, strategies):
    """
    Calculate losses for various cyclization strategies on a linear peptide from a PDB file.

    Parameters:
    pdb_path (str): Path to the PDB file containing the linear peptide structure.
    strategies (list of str): List of cyclization strategies to compute losses for.
                              Options include "disulfide", "amide", "side_chain_amide",
                              "thioether", "ester", and "hydrazone".

    Returns:
    dict: Dictionary with strategy names as keys and computed loss values as values.
    """
    # Load PDB file using MDTraj
    traj = md.load(pdb_path)
    topology = traj.topology
    positions = traj.xyz[0] / scaling_factor  # Scale coordinates for compatibility
    ### NEED TO PROPERLY HANDLE THE SCALING FACTOR, BENJAMIN

    # Helper to extract atom positions by name
    def get_atom_position(residue, atom_name):
        atom = [a for a in residue.atoms if a.name == atom_name]
        if not atom:
            raise ValueError(f"Atom {atom_name} not found in residue {residue}.")
        return positions[atom[0].index]

    # Initialize a dictionary for losses
    losses = {}

    # Iterate over strategies
    for strategy in strategies:
        if strategy == "disulfide":
            # Find pairs of cysteine residues
            cysteines = [r for r in topology.residues if r.name == "CYS"]
            if len(cysteines) < 2:
                losses["disulfide"] = None  # Not applicable
                continue

            # Compute disulfide losses for all pairs
            total_loss = 0
            for i in range(len(cysteines)):
                for j in range(i + 1, len(cysteines)):
                    cys1 = cysteines[i]
                    cys2 = cysteines[j]
                    cys1_s = get_atom_position(cys1, "SG")
                    cys1_cb = get_atom_position(cys1, "CB")
                    cys2_s = get_atom_position(cys2, "SG")
                    cys2_cb = get_atom_position(cys2, "CB")
                    total_loss += disulfide_loss(cys1_s, cys1_cb, cys2_s, cys2_cb)
            losses["disulfide"] = total_loss

        elif strategy == "amide":
            # Use backbone atoms for the amide bond
            residues = [r for r in topology.residues]
            if len(residues) < 2:
                losses["amide"] = None  # Not applicable
                continue

            # Compute amide loss for adjacent residues
            total_loss = 0
            for i in range(len(residues) - 1):
                res1 = residues[i]
                res2 = residues[i + 1]
                c1 = get_atom_position(res1, "C")
                ca1 = get_atom_position(res1, "CA")
                n2 = get_atom_position(res2, "N")
                h2 = get_atom_position(res2, "H")  # Optional: May not exist in all PDBs
                total_loss += amide_loss(c1, ca1, n2, h2)
            losses["amide"] = total_loss

        elif strategy == "side_chain_amide":
            # Find residues with carboxyl and amine side chains
            carboxyl_residues = [r for r in topology.residues if r.name in ["ASP", "GLU"]]
            amine_residues = [r for r in topology.residues if r.name in ["LYS", "ORN"]]
            if not carboxyl_residues or not amine_residues:
                losses["side_chain_amide"] = None  # Not applicable
                continue

            # Compute side-chain amide losses for all combinations
            total_loss = 0
            for carboxyl in carboxyl_residues:
                for amine in amine_residues:
                    c_carboxyl = get_atom_position(carboxyl, "CG")
                    carboxyl_anchor = get_atom_position(carboxyl, "CB")
                    n_side_chain = get_atom_position(amine, "NZ")
                    side_chain_anchor = get_atom_position(amine, "CE")
                    total_loss += side_chain_amide_loss(
                        n_side_chain, c_carboxyl, side_chain_anchor, carboxyl_anchor
                    )
            losses["side_chain_amide"] = total_loss

        elif strategy == "thioether":
            # Add thioether logic here
            pass

        elif strategy == "ester":
            # Add ester logic here
            pass

        elif strategy == "hydrazone":
            # Add hydrazone logic here
            pass

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    return losses

def total_cyclization_loss():
    '''
    Calculate total loss accross all cyclizations.
    '''
    pass
