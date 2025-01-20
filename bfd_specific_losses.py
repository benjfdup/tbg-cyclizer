### Imports ###
from typing import Dict
from math import pi

import torch
from bfd_constituent_losses import bond_angle_loss, dihedral_angle_loss, distance_loss
from abc import ABC, abstractmethod

def calc_total_loss(distance_losses: torch.tensor, bond_angle_losses: torch.tensor, dihedral_losses: torch.tensor, 
                        use_bond_distances: bool, use_bond_angles: bool, use_dihedrals: bool, weights: dict):
    
        # weights encodes how much of each type of loss to weight.
        total_loss = torch.zeros_like(distance_losses)

        if use_bond_distances:
            total_loss += distance_losses * weights['bond_distances']
        
        if use_bond_angles:
            total_loss += bond_angle_losses * weights['bond_angles']
        
        if use_dihedrals:
            total_loss += dihedral_losses * weights['dihedral_angles']

        return total_loss


### Defining the chemically specific losses
def disulfide_loss(cys1_s: torch.Tensor, cys1_cb: torch.Tensor, 
                   cys2_s: torch.Tensor, cys2_cb: torch.Tensor, 
                   weights: dict,

                   target_distance=2.05, # Typical bond length in Å (?)
                   target_bond_angle=torch.deg2rad(torch.tensor(102.5)), # Typical bond angles in radians (?)
                   target_dihedral_angle=torch.deg2rad(torch.tensor(90.0)), # Typical bond angles in radians (?)
                   
                   distance_tolerance=0.2, 
                   angle_tolerance=0.1,
                   steepness=1.0,

                   use_bond_distances: bool= True, # whether or not bond distances are added to the final loss
                   use_bond_angles: bool= True, # whether or not bond angles are added to the final loss
                   use_dihedrals: bool= True, # whether or not dihedral losses are added to the final loss
                   ): #TODO: implement steepness
    """
    Compute the disulfide loss for a specific pair of cysteine residues.

    Parameters:
    ----------
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

    total_loss = calc_total_loss(distance_losses= dist_loss, 
                                 bond_angle_losses= angle1_loss + angle2_loss, 
                                 dihedral_losses= dihedral_loss, 
                                 use_bond_distances= use_bond_distances,
                                 use_bond_angles= use_bond_angles,
                                 use_dihedrals= use_dihedrals,
                                 weights= weights)

    return total_loss

def h2t_amide_loss(c1, ca1, n2, h2, 
                   target_distance=1.33,  # Typical C-N bond length in Å
                   target_bond_angle=torch.deg2rad(torch.tensor(120.0)),  # Typical bond angles in radians
                   target_dihedral_angle=torch.deg2rad(torch.tensor(0.0)),  # Planarity implies dihedral angle ~0
                   
                   distance_tolerance=0.1,
                   angle_tolerance=0.1, 
                   steepness=1.0,

                   use_bond_distances: bool = True, # whether or not bond distances are added to the final loss
                   use_bond_angles: bool=True, # whether or not bond angles are added to the final loss
                   use_dihedrals: bool=True, # whether or not dihedral losses are added to the final loss
                   ):  # Controls steepness of penalties
    """
    Compute the loss for forming an amide bond between the head and tail residues.

    Parameters:
    ----------
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
    -------
    torch.Tensor: Total loss for H2T amide bond, shape (batch_size).
    """
    # Compute individual losses

    dist_loss = distance_loss(c1, n2, target_distance, distance_tolerance)  # Shape: (batch_size)
    angle1_loss = bond_angle_loss(ca1, c1, n2, target_bond_angle, angle_tolerance)  # Shape: (batch_size)
    angle2_loss = bond_angle_loss(c1, n2, h2, target_bond_angle, angle_tolerance)  # Shape: (batch_size)
    dihedral_loss = dihedral_angle_loss(ca1, c1, n2, h2, target_dihedral_angle, angle_tolerance)  # Shape: (batch_size)

    # Combine all losses
    total_loss = dist_loss + angle1_loss + angle2_loss + dihedral_loss  # Shape: (batch_size, )

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
    -------------------
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
    dist_loss = distance_loss(sulfur_atom, carbon_atom, target_distance, distance_tolerance)  # Shape: (batch_size, )

    # Bond Angle Losses
    angle1_loss = bond_angle_loss(sulfur_anchor, sulfur_atom, carbon_atom, target_bond_angle, angle_tolerance)  # Shape: (batch_size, )
    angle2_loss = bond_angle_loss(sulfur_atom, carbon_atom, carbon_anchor, target_bond_angle, angle_tolerance)  # Shape: (batch_size, )

    # Dihedral Angle Loss (Planarity)
    dihedral_loss = dihedral_angle_loss(sulfur_anchor, sulfur_atom, carbon_atom, carbon_anchor, target_dihedral_angle, angle_tolerance)  # Shape: (batch_size, )

    # Combine all losses
    total_loss = dist_loss + angle1_loss + angle2_loss + dihedral_loss  # Shape: (batch_size, )

    return total_loss

def ester_loss(oxygen_hydroxyl, carbon_carboxyl, hydroxyl_anchor, carboxyl_anchor,
               target_distance=1.4,  # Typical O-C bond distance in Å
               target_bond_angle=torch.deg2rad(torch.tensor(120.0)),  # Ester bond angles
               target_dihedral_angle=torch.deg2rad(torch.tensor(15.0)),  # Ester bond dihedral angle (check...)
               distance_tolerance=0.1,  # No-penalty range for distances
               angle_tolerance=0.2,  # No-penalty range for angles
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
    dist_loss = distance_loss(oxygen_hydroxyl, carbon_carboxyl, target_distance, distance_tolerance)  # Shape: (batch_size, )

    # Bond Angle Losses
    angle1_loss = bond_angle_loss(hydroxyl_anchor, oxygen_hydroxyl, carbon_carboxyl, target_bond_angle, angle_tolerance)  # Shape: (batch_size, )
    angle2_loss = bond_angle_loss(oxygen_hydroxyl, carbon_carboxyl, carboxyl_anchor, target_bond_angle, angle_tolerance)  # Shape: (batch_size, )

    # Dihedral Angle Loss (Planarity)
    dihedral_loss = dihedral_angle_loss(hydroxyl_anchor, oxygen_hydroxyl, carbon_carboxyl, carboxyl_anchor, target_dihedral_angle, angle_tolerance)  # Shape: (batch_size, )

    # Combine all losses
    total_loss = dist_loss + angle1_loss + angle2_loss + dihedral_loss  # Shape: (batch_size, )

    return total_loss

def hydrazone_loss(nitrogen_hydrazine, carbon_carbonyl, hydrazine_anchor, carbonyl_anchor,
                   target_distance=1.28,  # Typical N=C bond distance in Å
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
    dist_loss = distance_loss(nitrogen_hydrazine, carbon_carbonyl, target_distance, distance_tolerance)  # Shape: (batch_size, )

    # Bond Angle Losses
    angle1_loss = bond_angle_loss(hydrazine_anchor, nitrogen_hydrazine, carbon_carbonyl, target_bond_angle, angle_tolerance)  # Shape: (batch_size, )
    angle2_loss = bond_angle_loss(nitrogen_hydrazine, carbon_carbonyl, carbonyl_anchor, target_bond_angle, angle_tolerance)  # Shape: (batch_size, )

    # Dihedral Angle Loss (Planarity)
    dihedral_loss = dihedral_angle_loss(hydrazine_anchor, nitrogen_hydrazine, carbon_carbonyl, carbonyl_anchor, target_dihedral_angle, angle_tolerance)  # Shape: (batch_size, )

    # Combine all losses
    total_loss = dist_loss + angle1_loss + angle2_loss + dihedral_loss  # Shape: (batch_size, )

    return total_loss

def bis_thioether_loss(sulfur1, carbon1, sulfur2, carbon2, 
                       target_distance=1.82,  # Typical S-C bond distance in Å
                       target_bond_angle=torch.deg2rad(torch.tensor(109.5)),  # Ideal tetrahedral bond angle
                       target_dihedral_angle=torch.deg2rad(torch.tensor(180.0)),  # Planarity of macrocycle
                       distance_tolerance=0.2,  # Acceptable tolerance for bond distances
                       angle_tolerance=0.1,  # Acceptable tolerance for bond angles
                       steepness=1.0):  # Controls steepness of penalties
    """
    Computes the loss for forming bis-thioether macrocycles.

    This loss evaluates the geometric constraints required for proper bis-thioether macrocycle 
    formation, which involves sulfur atoms (e.g., from cysteine) forming bonds with carbon atoms 
    on another residue's side chain.

    Parameters:
    ----------
    sulfur1 (torch.Tensor): 
        Position of the first sulfur atom (e.g., SG of cysteine), shape (batch_size, 3).
        
    carbon1 (torch.Tensor): 
        Position of the first carbon atom bonded to the first sulfur, shape (batch_size, 3).

    sulfur2 (torch.Tensor): 
        Position of the second sulfur atom, shape (batch_size, 3).

    carbon2 (torch.Tensor): 
        Position of the second carbon atom bonded to the second sulfur, shape (batch_size, 3).

    target_distance (float): 
        Ideal S-C bond distance in Å (default: 2.0).

    target_bond_angle (float): 
        Ideal bond angle around the S-C bonds in radians (default: 109.5°).

    target_dihedral_angle (float): 
        Ideal dihedral angle for the macrocycle in radians (default: 180°).

    distance_tolerance (float): 
        Acceptable deviation range for bond distances, below which no penalty is applied.

    angle_tolerance (float): 
        Acceptable deviation range for bond angles, below which no penalty is applied.

    steepness (float): 
        Controls the steepness of the penalties for deviations from the ideal geometry.

    Returns:
    -------
    torch.Tensor:
        Total loss for bis-thioether macrocycle across all batches, shape (batch_size).

    Applicability:
    --------------
    This loss is used for enforcing bis-thioether macrocycle formation in peptides, particularly 
    in stabilizing helix capping or other macrocyclic designs. Suitable residues typically involve 
    cysteines forming covalent bonds with alkyl groups.
    """

    # Distance Losses for S-C bonds
    dist_loss1 = distance_loss(sulfur1, carbon1, target_distance, distance_tolerance)  # Shape: (batch_size, )
    dist_loss2 = distance_loss(sulfur2, carbon2, target_distance, distance_tolerance)  # Shape: (batch_size, )

    # Bond Angle Losses
    angle1_loss = bond_angle_loss(carbon1, sulfur1, carbon2, target_bond_angle, angle_tolerance)  # Shape: (batch_size, )
    angle2_loss = bond_angle_loss(carbon2, sulfur2, carbon1, target_bond_angle, angle_tolerance)  # Shape: (batch_size, )

    # Dihedral Angle Loss for macrocycle planarity
    dihedral_loss = dihedral_angle_loss(carbon1, sulfur1, sulfur2, carbon2, target_dihedral_angle, angle_tolerance)  # Shape: (batch_size, )

    # Combine all losses
    total_loss = dist_loss1 + dist_loss2 + angle1_loss + angle2_loss + dihedral_loss  # Shape: (batch_size, )

    return total_loss

def special_bis_thioether_loss(sulfur1, carbon1, sulfur2, carbon2, aromatic_c1, aromatic_c7,
                               target_distance_sc=1.82,  # S-C bond distance (Å)
                               target_distance_cc=1.39,  # Aromatic C-C bond distance (Å)
                               target_bond_angle=torch.deg2rad(torch.tensor(109.5)),  # Tetrahedral angle (sp3)
                               target_aromatic_angle=torch.deg2rad(torch.tensor(120.0)),  # Aromatic angle (sp2)
                               target_dihedral_angle=torch.deg2rad(torch.tensor(180.0)),  # Planarity
                               distance_tolerance=0.2, angle_tolerance=0.1, steepness=1.0):
    """
    Computes the loss for the BEN bis-thioether macrocycle.

    Parameters:
    ----------
    sulfur1, sulfur2: torch.Tensor
        Positions of the sulfur atoms in the BEN linker, shape (batch_size, 3).
    carbon1, carbon2: torch.Tensor
        Positions of the carbons bonded to sulfur atoms, shape (batch_size, 3).
    aromatic_c1, aromatic_c7: torch.Tensor
        Positions of aromatic carbons in the BEN linker, shape (batch_size, 3).

    Returns:
    -------
    torch.Tensor
        Total loss for the BEN bis-thioether linker, shape (batch_size).
    """
    # S-C Distance Losses
    dist_loss1 = distance_loss(sulfur1, carbon1, target_distance_sc, distance_tolerance)  # S1-C1
    dist_loss2 = distance_loss(sulfur2, carbon2, target_distance_sc, distance_tolerance)  # S2-C7

    # Aromatic C-C Distance Loss
    aromatic_dist_loss = distance_loss(aromatic_c1, aromatic_c7, target_distance_cc, distance_tolerance)  # C1-C7

    # Bond Angles (sp3 around sulfur)
    angle1_loss = bond_angle_loss(carbon1, sulfur1, carbon2, target_bond_angle, angle_tolerance)  # C1-S1-C2
    angle2_loss = bond_angle_loss(sulfur1, carbon1, aromatic_c1, target_aromatic_angle, angle_tolerance)  # S1-C1-C7
    angle3_loss = bond_angle_loss(sulfur2, carbon2, aromatic_c7, target_aromatic_angle, angle_tolerance)  # S2-C7-C1

    # Dihedral Angles (planarity of macrocycle)
    dihedral_loss = dihedral_angle_loss(carbon1, sulfur1, sulfur2, carbon2, target_dihedral_angle, angle_tolerance)  # Macrocycle planarity

    # Combine Losses
    total_loss = (dist_loss1 + dist_loss2 +
                  aromatic_dist_loss +
                  angle1_loss + angle2_loss + angle3_loss +
                  dihedral_loss)

    return total_loss