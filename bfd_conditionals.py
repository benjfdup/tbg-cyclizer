### Imports ###
import torch
import numpy as np
import mdtraj as md

from bfd_constituent_losses import bond_angle_loss, dihedral_angle_loss, soft_min, distance_loss
from bfd_constants import *

# TODO: need to add devices to the default tensor values (which are used when no other value is specified)

def precompute_atom_indices(residues, atom_names):
    """
    Precomputes atom indices for specified atom names across residues.
    """
    indices = {}
    for residue in residues:
        for atom_name in atom_names:
            atom = next((a for a in residue.atoms if a.name == atom_name), None)
            if atom:
                indices[(residue.index, atom_name)] = atom.index
    return indices

### Specific Losses:

def disulfide_loss(cys1_s, cys1_cb, cys2_s, cys2_cb, 
                   target_distance=2.05, # Typical bond length in Å (?)
                   target_bond_angle=torch.deg2rad(torch.tensor(110.0)), # Typical bond angles in radians (?)
                   target_dihedral_angle=torch.deg2rad(torch.tensor(90.0)), # Typical bond angles in radians (?)
                   distance_tolerance=0.2, 
                   angle_tolerance=0.1, 
                   steepness=1.0): #TODO: implement steepness
    """
    Compute the disulfide loss for a specific pair of cysteine residues.

    Parameters:
    cys1_s (torch.Tensor): Sulfur position of the first cysteine, shape (batch_size, 3).
    cys1_cb (torch.Tensor): Beta carbon position of the first cysteine, shape (batch_size, 3).
    cys2_s (torch.Tensor): Sulfur position of the second cysteine, shape (batch_size, 3).
    cys2_cb (torch.Tensor): Beta carbon position of the second cysteine, shape (batch_size, 3).

    Returns:
    torch.Tensor: Total loss for disulfide bond, shape (batch_size).
    """
    # Compute individual losses
    ### TODO: verify these with Alex & Google.
    dist_loss = distance_loss(cys1_s, cys2_s, target_distance, distance_tolerance)  # Shape: (batch_size)
    angle1_loss = bond_angle_loss(cys1_cb, cys1_s, cys2_s, target_bond_angle, angle_tolerance)  # Shape: (batch_size)
    angle2_loss = bond_angle_loss(cys1_s, cys2_s, cys2_cb, target_bond_angle, angle_tolerance)  # Shape: (batch_size)
    dihedral_loss = dihedral_angle_loss(cys1_cb, cys1_s, cys2_s, cys2_cb, target_dihedral_angle, angle_tolerance)  # Shape: (batch_size)

    # Combine all losses
    total_loss = dist_loss + angle1_loss + angle2_loss + dihedral_loss  # Shape: (batch_size)

    return total_loss

def h2t_amide_loss(c1, ca1, n2, h2, 
                   target_distance=1.33,  # Typical C-N bond length in Å
                   target_bond_angle=torch.deg2rad(torch.tensor(120.0)),  # Typical bond angles in radians
                   target_dihedral_angle=torch.deg2rad(torch.tensor(0.0)),  # Planarity implies dihedral angle ~0
                   distance_tolerance=0.1,
                   angle_tolerance=0.1, 
                   steepness=1.0):  # Controls steepness of penalties
    """
    Compute the loss for forming an amide bond between the head and tail residues.

    Parameters:
    c1 (torch.Tensor): Carbonyl carbon atom of the first residue, shape (batch_size, 3).
    ca1 (torch.Tensor): Alpha carbon atom of the first residue, shape (batch_size, 3).
    n2 (torch.Tensor): Amide nitrogen atom of the second residue, shape (batch_size, 3).
    h2 (torch.Tensor): Hydrogen attached to the amide nitrogen of the second residue, shape (batch_size, 3).
    target_distance (float): Ideal C-N bond distance in Å.
    target_bond_angle (float): Ideal bond angle in radians.
    target_dihedral_angle (float): Ideal dihedral angle in radians.
    distance_tolerance (float): Acceptable deviation in distance without penalty.
    angle_tolerance (float): Acceptable deviation in angles without penalty.
    steepness (float): Controls steepness of penalties.

    Returns:
    torch.Tensor: Total loss for H2T amide bond, shape (batch_size).
    """
    # Compute individual losses
    dist_loss = distance_loss(c1, n2, target_distance, distance_tolerance)  # Shape: (batch_size)
    angle1_loss = bond_angle_loss(ca1, c1, n2, target_bond_angle, angle_tolerance)  # Shape: (batch_size)
    angle2_loss = bond_angle_loss(c1, n2, h2, target_bond_angle, angle_tolerance)  # Shape: (batch_size)
    dihedral_loss = dihedral_angle_loss(ca1, c1, n2, h2, target_dihedral_angle, angle_tolerance)  # Shape: (batch_size)

    # Combine all losses
    total_loss = dist_loss + angle1_loss + angle2_loss + dihedral_loss  # Shape: (batch_size)

    return total_loss

def side_chain_amide_loss(n_side_chain, c_carboxyl, side_chain_anchor, carboxyl_anchor,
                          target_distance=1.33,  # Typical amide bond distance in Å
                          target_bond_angle=torch.deg2rad(torch.tensor(120.0)),  # Typical amide bond angle
                          target_dihedral_angle=torch.deg2rad(torch.tensor(0.0)),  # Planarity of amide bond
                          distance_tolerance=0.1,  # No-penalty range for distances
                          angle_tolerance=0.1,  # No-penalty range for angles
                          steepness=1.0):  # Factor controlling penalty steepness
    """
    Computes the loss for forming side-chain amide bonds in a parallelized way over batches.

    This loss is applicable to residues capable of forming amide bonds between a side-chain 
    nitrogen (e.g., lysine or ornithine) and a carboxyl group (e.g., aspartic acid or glutamic acid). 
    The formation of these bonds involves both distance and angular constraints, which are critical 
    for maintaining bond geometry and molecular planarity.

    Parameters:
    ----------
    n_side_chain (torch.Tensor): 
        Positions of the side-chain nitrogen atoms, shape (batch_size, 3).
        Typically corresponds to nitrogen atoms from lysine (NZ) or ornithine.
        
    c_carboxyl (torch.Tensor): 
        Positions of the carboxyl group carbon atoms, shape (batch_size, 3).
        Typically corresponds to carbon atoms from aspartic acid (CG) or glutamic acid (CD).

    side_chain_anchor (torch.Tensor): 
        Positions of the anchor atoms connected to the side-chain nitrogen, shape (batch_size, 3).
        For lysine or ornithine, this would be the CE atom.

    carboxyl_anchor (torch.Tensor): 
        Positions of the anchor atoms connected to the carboxyl group carbon, shape (batch_size, 3).
        For aspartic acid or glutamic acid, this would be the CB atom.
    
    Optional Parameters:
    ----------
    target_distance (float): 
        Ideal bond distance between the side-chain nitrogen and the carboxyl carbon in Ångstroms (default: 1.33).

    target_bond_angle (float): 
        Ideal bond angle around the amide bond in radians (default: 120° or π/3 radians).

    target_dihedral_angle (float): 
        Ideal dihedral angle for planarity of the amide bond in radians (default: 0 radians).

    distance_tolerance (float): 
        The acceptable deviation range for bond distances, below which no penalty is applied.

    angle_tolerance (float): 
        The acceptable deviation range for bond angles, below which no penalty is applied.

    steepness (float): 
        Controls the steepness of the penalties for deviations from the ideal geometry (not implemented yet)

    Returns:
    -------
    torch.Tensor:
        Total loss for side-chain amide bonds across all batches, shape (batch_size).

    Applicability:
    --------------
    This loss is used when modeling or enforcing side-chain amide bond formation in molecules, 
    such as:
      - Covalent bonding during peptide cyclization or cross-linking.
      - Amide formation in protein engineering or drug design.
    Suitable residue pairs include:
      - Lysine (NZ) or ornithine forming bonds with the carboxyl groups of aspartic acid or glutamic acid.
    It is particularly relevant for studying side-chain-specific interactions or designing 
    engineered peptide structures.

    Notes:
    ------
    This function is batch-friendly and supports parallel computation across all input batches 
    to ensure efficiency in large datasets or simulations.
    """

    # Distance Loss (N-C bond)
    dist_loss = distance_loss(n_side_chain, c_carboxyl, target_distance, distance_tolerance)  # Shape: (batch_size)

    # Bond Angle Losses
    angle1_loss = bond_angle_loss(side_chain_anchor, n_side_chain, c_carboxyl, target_bond_angle, angle_tolerance)  # Shape: (batch_size)
    angle2_loss = bond_angle_loss(n_side_chain, c_carboxyl, carboxyl_anchor, target_bond_angle, angle_tolerance)  # Shape: (batch_size)

    # Dihedral Angle Loss (Planarity)
    dihedral_loss = dihedral_angle_loss(side_chain_anchor, n_side_chain, c_carboxyl, carboxyl_anchor, target_dihedral_angle, angle_tolerance)  # Shape: (batch_size)

    # Combine all losses
    total_loss = dist_loss + angle1_loss + angle2_loss + dihedral_loss  # Shape: (batch_size)

    return total_loss

def thioether_loss(sulfur_atom, carbon_atom, sulfur_anchor, carbon_anchor,
                   target_distance=1.8,  # Typical S-C bond distance in Å
                   target_bond_angle=torch.deg2rad(torch.tensor(109.0)),  # Bond angle for sp3 hybridized atoms
                   target_dihedral_angle=torch.deg2rad(torch.tensor(90.0)),  # Typical dihedral angle
                   distance_tolerance=0.2,  # No-penalty range for distances
                   angle_tolerance=0.1,  # No-penalty range for angles
                   steepness=1.0):  # Factor controlling penalty steepness
    """
    Computes the loss for forming thioether bonds in a parallelized way over batches.

    This loss is applicable to residues capable of forming thioether bonds, such as cysteine (S) 
    and methionine (SD) interacting with carbon atoms from another residue's side chain. 
    It evaluates the geometric constraints essential for proper thioether bond formation.

    Parameters:
    ----------
    sulfur_atom (torch.Tensor): 
        Positions of sulfur atoms, shape (batch_size, 3).
        Typically corresponds to sulfur atoms from cysteine (SG) or methionine (SD).
    
    carbon_atom (torch.Tensor): 
        Positions of carbon atoms, shape (batch_size, 3).
        Typically corresponds to carbon atoms from another residue's side chain (e.g., CG).

    sulfur_anchor (torch.Tensor): 
        Positions of anchor atoms connected to the sulfur, shape (batch_size, 3).
        For cysteine, this would be CB; for methionine, this would be CE.

    carbon_anchor (torch.Tensor): 
        Positions of anchor atoms connected to the carbon, shape (batch_size, 3).
        Typically CB or CG depending on the residue.

    Optional Parameters:
    ----------
    target_distance (float): 
        Ideal bond distance between sulfur and carbon in Ångstroms (default: 1.8).

    target_bond_angle (float): 
        Ideal bond angle around the thioether bond in radians (default: 109° or π/3 radians).

    target_dihedral_angle (float): 
        Ideal dihedral angle in radians (default: 90° or π/2 radians).

    distance_tolerance (float): 
        Acceptable deviation range for bond distances, below which no penalty is applied.

    angle_tolerance (float): 
        Acceptable deviation range for bond angles, below which no penalty is applied.

    steepness (float): 
        Controls the steepness of penalties for deviations from the ideal geometry.

    Returns:
    -------
    torch.Tensor:
        Total loss for thioether bonds across all batches, shape (batch_size).

    Applicability:
    --------------
    This loss is used when modeling or enforcing thioether bond formation in molecules, 
    such as:
      - Cross-linking involving cysteine (e.g., in lipidation or glycosylation).
      - Structural stability of proteins involving sulfur-carbon interactions.
    Suitable residue pairs include:
      - Cysteine (SG) or methionine (SD) forming bonds with carbon atoms in other residues.
    It is particularly relevant for studying sulfur-specific interactions or designing 
    engineered peptide structures.

    Notes:
    ------
    This function is batch-friendly and supports parallel computation across all input batches 
    to ensure efficiency in large datasets or simulations.
    """

    # Distance Loss (S-C bond)
    dist_loss = distance_loss(sulfur_atom, carbon_atom, target_distance, distance_tolerance)  # Shape: (batch_size)

    # Bond Angle Losses
    angle1_loss = bond_angle_loss(sulfur_anchor, sulfur_atom, carbon_atom, target_bond_angle, angle_tolerance)  # Shape: (batch_size)
    angle2_loss = bond_angle_loss(sulfur_atom, carbon_atom, carbon_anchor, target_bond_angle, angle_tolerance)  # Shape: (batch_size)

    # Dihedral Angle Loss (Planarity)
    dihedral_loss = dihedral_angle_loss(sulfur_anchor, sulfur_atom, carbon_atom, carbon_anchor, target_dihedral_angle, angle_tolerance)  # Shape: (batch_size)

    # Combine all losses
    total_loss = dist_loss + angle1_loss + angle2_loss + dihedral_loss  # Shape: (batch_size)

    return total_loss

def ester_loss(oxygen_hydroxyl, carbon_carboxyl, hydroxyl_anchor, carboxyl_anchor,
               target_distance=1.4,  # Typical O-C bond distance in Å
               target_bond_angle=torch.deg2rad(torch.tensor(120.0)),  # Ester bond angles
               target_dihedral_angle=torch.deg2rad(torch.tensor(90)),  # Ester bond dihedral angle (check...)
               distance_tolerance=0.1,  # No-penalty range for distances
               angle_tolerance=0.1,  # No-penalty range for angles
               steepness=1.0):  # Factor controlling penalty steepness
    """
    Computes the loss for forming ester bonds in a parallelized way over batches.

    This loss is applicable to residues capable of forming ester bonds, such as hydroxyl groups 
    (e.g., serine or threonine) interacting with carboxyl groups (e.g., aspartic acid or glutamic acid). 
    The formation of these bonds involves both distance and angular constraints, which are critical 
    for maintaining bond geometry and molecular planarity.

    Parameters:
    ----------
    oxygen_hydroxyl (torch.Tensor): 
        Positions of oxygen atoms from hydroxyl groups, shape (batch_size, 3).
        Typically corresponds to oxygen atoms from serine (OG) or threonine.

    carbon_carboxyl (torch.Tensor): 
        Positions of carbon atoms from carboxyl groups, shape (batch_size, 3).
        Typically corresponds to carbon atoms from aspartic acid (CG) or glutamic acid (CD).

    hydroxyl_anchor (torch.Tensor): 
        Positions of anchor atoms connected to the hydroxyl oxygen, shape (batch_size, 3).
        For serine or threonine, this would be the CB atom.

    carboxyl_anchor (torch.Tensor): 
        Positions of anchor atoms connected to the carboxyl carbon, shape (batch_size, 3).
        For aspartic acid or glutamic acid, this would be the CB atom.

    Optional Parameters:
    ----------
    target_distance (float): 
        Ideal bond distance between hydroxyl oxygen and carboxyl carbon in Ångstroms (default: 1.4).

    target_bond_angle (float): 
        Ideal bond angle around the ester bond in radians (default: 120° or π/3 radians).

    target_dihedral_angle (float): 
        Ideal dihedral angle for planarity of the ester bond in radians (default: 0 radians).

    distance_tolerance (float): 
        Acceptable deviation range for bond distances, below which no penalty is applied.

    angle_tolerance (float): 
        Acceptable deviation range for bond angles, below which no penalty is applied.

    steepness (float): 
        Controls the steepness of penalties for deviations from the ideal geometry.

    Returns:
    -------
    torch.Tensor:
        Total loss for ester bonds across all batches, shape (batch_size).

    Applicability:
    --------------
    This loss is used when modeling or enforcing ester bond formation in molecules, 
    such as:
      - Covalent bonding during peptide cyclization or cross-linking.
      - Ester formation in protein engineering or drug design.
    Suitable residue pairs include:
      - Hydroxyl groups (OG) of serine or threonine forming bonds with carboxyl groups of aspartic acid or glutamic acid.
    It is particularly relevant for studying ester-specific interactions or designing 
    engineered peptide structures.

    Notes:
    ------
    This function is batch-friendly and supports parallel computation across all input batches 
    to ensure efficiency in large datasets or simulations.
    """

    # Distance Loss (O-C bond)
    dist_loss = distance_loss(oxygen_hydroxyl, carbon_carboxyl, target_distance, distance_tolerance)  # Shape: (batch_size)

    # Bond Angle Losses
    angle1_loss = bond_angle_loss(hydroxyl_anchor, oxygen_hydroxyl, carbon_carboxyl, target_bond_angle, angle_tolerance)  # Shape: (batch_size)
    angle2_loss = bond_angle_loss(oxygen_hydroxyl, carbon_carboxyl, carboxyl_anchor, target_bond_angle, angle_tolerance)  # Shape: (batch_size)

    # Dihedral Angle Loss (Planarity)
    dihedral_loss = dihedral_angle_loss(hydroxyl_anchor, oxygen_hydroxyl, carbon_carboxyl, carboxyl_anchor, target_dihedral_angle, angle_tolerance)  # Shape: (batch_size)

    # Combine all losses
    total_loss = dist_loss + angle1_loss + angle2_loss + dihedral_loss  # Shape: (batch_size)

    return total_loss

def hydrazone_loss(nitrogen_hydrazine, carbon_carbonyl, hydrazine_anchor, carbonyl_anchor,
                   target_distance=1.45,  # Typical N=C bond distance in Å
                   target_bond_angle=torch.deg2rad(torch.tensor(120.0)),  # Typical hydrazone bond angles
                   target_dihedral_angle=torch.deg2rad(torch.tensor(180.0)),  # Planarity of hydrazone bond
                   distance_tolerance=0.1,  # No-penalty range for distances
                   angle_tolerance=0.1,  # No-penalty range for angles
                   steepness=1.0):  # Factor controlling penalty steepness
    """
    Computes the loss for forming hydrazone bonds in a parallelized way over batches.

    This loss models the formation of hydrazone bonds, typically found in chemical crosslinking 
    reactions between hydrazine derivatives and carbonyl groups. These bonds are often planar 
    with specific geometric constraints.

    Parameters:
    ----------
    nitrogen_hydrazine (torch.Tensor): 
        Positions of the nitrogen atoms in the hydrazine group, shape (batch_size, 3).
        
    carbon_carbonyl (torch.Tensor): 
        Positions of the carbon atoms in the carbonyl group, shape (batch_size, 3).

    hydrazine_anchor (torch.Tensor): 
        Positions of the anchor atoms connected to the hydrazine nitrogen, shape (batch_size, 3).

    carbonyl_anchor (torch.Tensor): 
        Positions of the anchor atoms connected to the carbonyl carbon, shape (batch_size, 3).
    
    Optional Parameters:
    ----------
    target_distance (float): 
        Ideal bond distance between nitrogen and carbon in Å (default: 1.45).

    target_bond_angle (float): 
        Ideal bond angle around the hydrazone bond in radians (default: 120° or π/3 radians).

    target_dihedral_angle (float): 
        Ideal dihedral angle for hydrazone bond planarity in radians (default: 180° or π radians).

    distance_tolerance (float): 
        The acceptable deviation range for bond distances, below which no penalty is applied.

    angle_tolerance (float): 
        The acceptable deviation range for bond angles, below which no penalty is applied.

    steepness (float): 
        Controls the steepness of the penalties for deviations from the ideal geometry.

    Returns:
    -------
    torch.Tensor:
        Total loss for hydrazone bonds across all batches, shape (batch_size).

    Applicability:
    --------------
    This loss is applicable in modeling hydrazone bond formation, such as in:
      - Chemical crosslinking reactions in bioconjugation.
      - Protein engineering and small molecule design.
    """

    # Distance Loss (N=C bond)
    dist_loss = distance_loss(nitrogen_hydrazine, carbon_carbonyl, target_distance, distance_tolerance)  # Shape: (batch_size)

    # Bond Angle Losses
    angle1_loss = bond_angle_loss(hydrazine_anchor, nitrogen_hydrazine, carbon_carbonyl, target_bond_angle, angle_tolerance)  # Shape: (batch_size)
    angle2_loss = bond_angle_loss(nitrogen_hydrazine, carbon_carbonyl, carbonyl_anchor, target_bond_angle, angle_tolerance)  # Shape: (batch_size)

    # Dihedral Angle Loss (Planarity)
    dihedral_loss = dihedral_angle_loss(hydrazine_anchor, nitrogen_hydrazine, carbon_carbonyl, carbonyl_anchor, target_dihedral_angle, angle_tolerance)  # Shape: (batch_size)

    # Combine all losses
    total_loss = dist_loss + angle1_loss + angle2_loss + dihedral_loss  # Shape: (batch_size)

    return total_loss

### REVIEW AND FIX WHERE NEEDED...

def initialize_cyclization_loss(pdb_path, 
                                strategies=None,  
                                alpha=-10,
                                steepnesses=None,):
    """
    Initialize the cyclization loss function for a peptide, returning a closure
    for efficient repeated evaluations with new positions.

    Parameters:
    ----------
    pdb_path (str): 
        Path to the PDB file containing the peptide structure.

    strategies (list of str, optional): 
        Cyclization strategies to compute losses for.
        Options include "disulfide", "amide", "side_chain_amide",
        "thioether", "ester", "hydrazone", "h2t". 
        Default includes all strategies.

    steepnesses (dict of float, optional): 
        Dictionary mapping strategies to their steepness values. If not provided, 
        default steepness values will be used for each strategy.

    alpha (float): 
        Exponent for the soft_min function. Higher magnitude makes it closer to the true minimum.

    Returns:
    -------
    function: 
        A closure that computes the total loss given positions.
    """
    if strategies is None:
        strategies = ['disulfide', 'amide', 'side_chain_amide', 'thioether', 'ester', 'hydrazone', 'h2t']

    # Load PDB file and topology
    traj = md.load(pdb_path)
    topology = traj.topology
    residue_list = list(topology.residues)
    
    bonding_atoms = ["SG", "CB", "C", "CA", "N", "H", "NZ", "CE", "CG", "OG", "SD", "NE", "CD"]
    all_atom_indices = precompute_atom_indices(residue_list, bonding_atoms)

    def get_atom_indices(residue, atom_names):
        return [all_atom_indices[(residue.index, atom_name)] for atom_name in atom_names]

    # Define sub-loss functions based on selected strategies
    loss_functions = []

    # Helper to convert atom positions into tensors
    def extract_positions(indices, positions):
        return torch.stack([positions[idx] for idx in indices], dim=0)

    if "disulfide" in strategies:
        cysteines = [r for r in residue_list if r.name == "CYS"]
        if len(cysteines) > 1:
            sulfur_positions = []
            beta_positions = []
            for cys in cysteines:
                sulfur_indices = get_atom_indices(cys, ["SG"])
                beta_indices = get_atom_indices(cys, ["CB"])
                sulfur_positions.append(sulfur_indices[0])
                beta_positions.append(beta_indices[0])
            sulfur_positions = torch.tensor(sulfur_positions)
            beta_positions = torch.tensor(beta_positions)

            loss_functions.append(lambda pos: disulfide_loss(
                pos[sulfur_positions], pos[beta_positions]
            ))

    if "amide" in strategies:
        min_residue_distance = 3
        amide_pairs = [
            (residue_list[i], residue_list[j]) for i in range(len(residue_list))
            for j in range(i + min_residue_distance, len(residue_list))
        ]
        if amide_pairs:
            carbon_positions = []
            nitrogen_positions = []
            for pair in amide_pairs:
                carbon_positions.append(get_atom_indices(pair[0], ["C", "CA"]))
                nitrogen_positions.append(get_atom_indices(pair[1], ["N", "H"]))
            loss_functions.append(lambda pos: side_chain_amide_loss(
                extract_positions(carbon_positions, pos),
                extract_positions(nitrogen_positions, pos)
            ))

    if "side_chain_amide" in strategies:
        carboxyl_residues = [r for r in residue_list if r.name in ["ASP", "GLU"]]
        amine_residues = [r for r in residue_list if r.name in ["LYS", "ORN"]]
        if carboxyl_residues and amine_residues:
            carboxyl_positions = []
            amine_positions = []
            for carboxyl in carboxyl_residues:
                carboxyl_positions.append(get_atom_indices(carboxyl, ["CG", "CB"]))
            for amine in amine_residues:
                amine_positions.append(get_atom_indices(amine, ["NZ", "CE"]))
            loss_functions.append(lambda pos: side_chain_amide_loss(
                extract_positions(amine_positions, pos),
                extract_positions(carboxyl_positions, pos)
            ))

    # Add more strategies similarly...
    # (e.g., thioether, ester, hydrazone, h2t)

    # Closure for loss computation
    def cyclization_loss(positions): # this will be very fast, regardless, compared to the many
        # position updates that have to occur during the flow matching EGNN passings.

        """
        Compute the cyclization loss for the given atom positions.

        Parameters:
        ----------
        positions (torch.Tensor): Tensor of shape (N_atoms, 3) representing the atomic positions.

        Returns:
        -------
        torch.Tensor: Total cyclization loss as a scalar.
        """
        batched_losses = torch.stack([loss(positions) for loss in loss_functions], dim=0)

        return soft_min(batched_losses, alpha=alpha).sum()

    return cyclization_loss
