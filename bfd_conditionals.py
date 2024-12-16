### Imports ###
import warnings
import math

import torch
import mdtraj as md

from bfd_constituent_losses import soft_min
from bfd_constants import *
from bfd_specific_losses import *

### This file contains all of the code and organizational functions necessary to easily handle loss implementation.
# TODO: need to add devices to the default tensor values (which are used when no other value is specified)

def gaussian_w_t(mu, s, coeff = 1):
    '''
    Generates a function thich takes a tensor of shape (1, ) as an input and outputs
    a number between 0 & 1, representing the loss gradient coefficient for that particular time.

    Args:
    ----
    mu (float): the temporal location of the maximum. Must be between 0 & 1
    s (float): the width of the maximum. Must be > 0

    Returns:
    -------
    w_t (torch.Tensor): the loss coefficient for that particular moment in time.
    '''

    if not (0 < coeff <= 1):
        warnings.warn(f"""Warning. The output function will produce values out of the acceptable (0, 1]
                      range. coeff={coeff} should be in (0, 1]""")

    w_t = lambda t: (coeff * torch.exp(-0.5 * ((t - mu) / s)**2)).clamp(0, 1)

    w_t.__doc__ = f'''
    Returns the loss coefficient for the specified moment in time. This family is drawn from
    a "pseudo-gaussian" with params mu: {mu}, s: {s}, coeff: {coeff}.

    Args:
    ----
    t (float): a float between 0 & 1 representing the moment in time of evaluation.

    Returns:
    -------
    w (float): a float between 0 & 1 representing the loss coefficient at that moment.

    '''

    return w_t

def maxwell_boltzmann_w_t(a):
    '''
    this function is normalized, save when it is out of (0, 1].
    need to implement relevant warnings.
    '''
    pi = torch.tensor(math.pi)

    w_t = lambda t: torch.sqrt(2 / pi) * (t**2 / a**3) * torch.exp(-t**2 / (2 * a**2))

    return w_t

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

### Class to handle cyclic loss condition
class cyclization_loss_handler(): #TODO: implement steepnesses
    def __init__(self, pdb_path,
                 alpha = -10, 
                 device = None,
                 strategies=['disulfide', 'amide', 'side_chain_amide', 
                             'thioether', 'ester', 'hydrazone', 'h2t'],):
        self._pdb_path = pdb_path
        self._alpha = alpha
        self._device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._strategies = strategies

        # should I make these private?
        self._loss_functions = []
        self._strategies_indices_pair_list = []
        self._cyclization_loss = None

        self._initialize_loss()

    def _initialize_loss(self):
        """
        Internal method to initialize the cyclization loss function for a peptide.
        """

        # Load PDB file and topology
        traj = md.load(self._pdb_path)
        topology = traj.topology
        residue_list = list(topology.residues)
        bonding_atoms = ["SG", "CB", "C", "CA", "N", "H", "NZ", "CE", "CG", "OG", "SD", "NE", "CD"]
        all_atom_indices = precompute_atom_indices(residue_list, bonding_atoms)

        def get_atom_indices(residue, atom_names):
            """
            Retrieves the atom indices for specified atom names in a given residue.

            Args:
                residue: A residue object from the topology containing atom information.
                atom_names (list of str): A list of atom names for which indices are required.

            Returns:
                list of int: A list of atom indices corresponding to the specified atom names
                in the given residue.
            """
            return [all_atom_indices[(residue.index, atom_name)] for atom_name in atom_names]

        # Define sub-loss functions based on selected strategies
        loss_functions = []
        
        strategies_indices_pair_list = []

        # Helper to convert atom positions into tensors
        def extract_positions(indices, positions): # needed?
            return torch.stack([positions[idx] for idx in indices], dim=0)

        if "disulfide" in self._strategies: 
            cysteines = [r for r in residue_list if r.name == "CYS"]
            if len(cysteines) > 1:
                for i, cys_1 in enumerate(cysteines):
                    for cys_2 in cysteines[i + 1:]:  # Avoid duplicate pairs
                        # Extract individual atom indices
                        sulfur_index_1 = get_atom_indices(cys_1, ["SG"])[0]
                        sulfur_index_2 = get_atom_indices(cys_2, ["SG"])[0]
                        beta_index_1 = get_atom_indices(cys_1, ["CB"])[0]
                        beta_index_2 = get_atom_indices(cys_2, ["CB"])[0]

                        # Add the loss function with captured arguments
                        loss_functions.append(
                            lambda pos, 
                            s1=sulfur_index_1, 
                            s2=sulfur_index_2, 
                            cb1=beta_index_1,
                            cb2=beta_index_2: disulfide_loss(
                                pos[:, s1, :].squeeze(),  # Sulfur of cysteine 1
                                pos[:, cb1, :].squeeze(), # Beta-carbon of cysteine 1
                                pos[:, s2, :].squeeze(),  # Sulfur of cysteine 2
                                pos[:, cb2, :].squeeze(), # Beta-carbon of cysteine 2
                            )
                        )

                        # Append strategy and amino acid indices to the list
                        strategies_indices_pair_list.append((
                            "disulfide", 
                            (cys_1.index, cys_2.index)
                        ))
                        
        if "side_chain_amide" in self._strategies:
            carboxyl_residues = [r for r in residue_list if r.name in ["ASP", "GLU"]]
            amine_residues = [r for r in residue_list if r.name in ["LYS", "ORN"]]
            if carboxyl_residues and amine_residues:
                for carboxyl in carboxyl_residues:
                    for amine in amine_residues:
                        # Extract individual atom indices
                        carboxyl_indices = get_atom_indices(carboxyl, ["CG", "CB"])
                        amine_indices = get_atom_indices(amine, ["NZ", "CE"])

                        # Add the loss function with captured arguments
                        loss_functions.append(
                            lambda pos, 
                            n_idx=amine_indices[0],  # Side-chain amine
                            c_idx=carboxyl_indices[0],  # Carboxyl group
                            a_anchor=amine_indices[1],  # Amine side-chain anchor
                            c_anchor=carboxyl_indices[1]:  # Carboxyl side-chain anchor
                            side_chain_amide_loss(
                                pos[:, n_idx, :].squeeze(), 
                                pos[:, c_idx, :].squeeze(),
                                pos[:, a_anchor, :].squeeze(), 
                                pos[:, c_anchor, :].squeeze(),
                            )
                        )

                        # Append strategy and amino acid indices to the list
                        strategies_indices_pair_list.append((
                            "side_chain_amide",
                            (carboxyl.index, amine.index)
                        ))

        if "thioether" in self._strategies:
            thiol_residues = [r for r in residue_list if r.name in ["CYS", "MET"]]
            alkyl_residues = [r for r in residue_list if r.name in ["LYS", "ORN", "ALA"]]  # Example residues with alkyl groups
            if thiol_residues and alkyl_residues:
                for thiol in thiol_residues:
                    for alkyl in alkyl_residues:
                        # Extract individual atom indices
                        thiol_indices = get_atom_indices(thiol, ["SG", "CB"])  # Sulfur and its anchor
                        alkyl_indices = get_atom_indices(alkyl, ["CE", "CB"])  # Alkyl group and its anchor

                        # Add the loss function with captured arguments
                        loss_functions.append(
                            lambda pos, 
                            s_atom=thiol_indices[0],  # Sulfur atom (thiol)
                            c_atom=alkyl_indices[0],  # Carbon atom (alkyl group)
                            s_anchor=thiol_indices[1],  # Sulfur anchor
                            c_anchor=alkyl_indices[1]:  # Carbon anchor
                            thioether_loss(
                                pos[:, s_atom, :].squeeze(), 
                                pos[:, c_atom, :].squeeze(),
                                pos[:, s_anchor, :].squeeze(), 
                                pos[:, c_anchor, :].squeeze(),
                            )
                        )

                        # Append strategy and amino acid indices to the list
                        strategies_indices_pair_list.append((
                            "thioether",
                            (thiol.index, alkyl.index)
                        ))

        if "ester" in self._strategies:
            serine_residues = [r for r in residue_list if r.name == "SER"]
            glutamate_residues = [r for r in residue_list if r.name == "GLU"]
            if serine_residues and glutamate_residues:
                for ser in serine_residues:
                    for glu in glutamate_residues:
                        # Extract individual atom indices
                        hydroxyl_index = get_atom_indices(ser, ["OG"])[0]  # Hydroxyl oxygen
                        carboxyl_index = get_atom_indices(glu, ["CD"])[0]  # Carboxyl carbon
                        ser_anchor = get_atom_indices(ser, ["CB"])[0]  # Serine anchor
                        glu_anchor = get_atom_indices(glu, ["CG"])[0]  # Glutamate anchor

                        # Add the loss function with captured arguments
                        loss_functions.append(
                            lambda pos, 
                            oxygen_hydroxyl=hydroxyl_index,  # Hydroxyl oxygen (serine)
                            carbon_carboxyl=carboxyl_index,  # Carboxyl carbon (glutamate)
                            hydroxyl_anchor=ser_anchor,  # Serine anchor atom
                            carboxyl_anchor=glu_anchor:  # Glutamate anchor atom
                            ester_loss(
                                pos[:, oxygen_hydroxyl, :].squeeze(), 
                                pos[:, carbon_carboxyl, :].squeeze(), 
                                pos[:, hydroxyl_anchor, :].squeeze(), 
                                pos[:, carboxyl_anchor, :].squeeze(),
                            )
                        )

                        # Append strategy and amino acid indices to the list
                        strategies_indices_pair_list.append((
                            "ester",
                            (ser.index, glu.index)
                        ))

        if "hydrazone" in self._strategies:
            lysine_residues = [r for r in residue_list if r.name == "LYS"]
            aldehyde_residues = [r for r in residue_list if r.name == "CHO"]  # Hypothetical example, non-canonical AA
            if lysine_residues and aldehyde_residues:
                for lys in lysine_residues:
                    for cho in aldehyde_residues:
                        # Extract individual atom indices
                        nitrogen_hydrazine = get_atom_indices(lys, ["NZ"])[0]  # Amine nitrogen (lysine)
                        carbon_carbonyl = get_atom_indices(cho, ["C"])[0]  # Carbonyl carbon (aldehyde)
                        hydrazine_anchor = get_atom_indices(lys, ["CE"])[0]  # Lysine anchor atom
                        carbonyl_anchor = get_atom_indices(cho, ["O"])[0]  # Aldehyde anchor atom

                        # Add the loss function with captured arguments
                        loss_functions.append(
                            lambda pos, 
                            n_hydrazine=nitrogen_hydrazine,  # Nitrogen hydrazine
                            c_carbonyl=carbon_carbonyl,  # Carbonyl carbon
                            h_anchor=hydrazine_anchor,  # Hydrazine anchor atom
                            c_anchor=carbonyl_anchor:  # Carbonyl anchor atom
                            hydrazone_loss(
                                pos[:, n_hydrazine, :].squeeze(), 
                                pos[:, c_carbonyl, :].squeeze(), 
                                pos[:, h_anchor, :].squeeze(), 
                                pos[:, c_anchor, :].squeeze(),
                            )
                        )

                        # Append strategy and amino acid indices to the list
                        strategies_indices_pair_list.append((
                            "hydrazone",
                            (lys.index, cho.index)
                        ))

        if "h2t" in self._strategies:
            # Identify terminal residues by checking their connectivity or sequence position
            terminal_residues = [res for res in residue_list if res.index == 0 or res.index == len(residue_list) - 1]
            if len(terminal_residues) == 2:
                # Extract individual atom indices for the N- and C-terminal residues
                c1_index = get_atom_indices(terminal_residues[1], ["C"])[0]  # C-terminal carbon
                ca1_index = get_atom_indices(terminal_residues[1], ["CA"])[0]  # C-terminal alpha-carbon
                n2_index = get_atom_indices(terminal_residues[0], ["N"])[0]  # N-terminal nitrogen
                h2_index = get_atom_indices(terminal_residues[0], ["H"])[0]  # Hydrogen attached to N-terminal nitrogen

                # Add the loss function with captured arguments
                loss_functions.append(
                    lambda pos, 
                    c1=c1_index,  # C-terminal carbon
                    ca1=ca1_index,  # C-terminal alpha-carbon
                    n2=n2_index,  # N-terminal nitrogen
                    h2=h2_index:  # Hydrogen attached to N-terminal nitrogen
                    h2t_amide_loss(
                        pos[:, c1, :].squeeze(), # this is, fundamentally, what you need to check.
                        pos[:, ca1, :].squeeze(), 
                        pos[:, n2, :].squeeze(), 
                        pos[:, h2, :].squeeze(),
                    )
                )

                # Append strategy and amino acid indices to the list
                strategies_indices_pair_list.append((
                    "h2t",
                    (terminal_residues[1].index, terminal_residues[0].index)
                ))

        # Add more strategies similarly...

        # Closure for loss computation
        def cyclization_loss(positions): # this will be very fast, regardless, compared to the many
            # position updates that have to occur during the flow matching EGNN passings.

            """
            Compute the cyclization loss for the given atom positions.

            Parameters:
            ----------
            positions (torch.Tensor): Tensor of shape (n_batch, n_atoms, 3) representing the atomic positions.

            Returns:
            -------
            torch.Tensor: Total cyclization loss as a scalar.
            """

            # Each loss must take an input of the shape (n_batch, n_atoms, 3) and output smthng of shape (n_batch, )
            # meaning the overall output of this function should eb of shape (n_loss, n_batch)...

            batched_losses = torch.stack([loss(positions) for loss in loss_functions], dim=1).squeeze() # check the required shape...
            # batched losses should be of shape (n_batches, n_losses)

            # soft_min expects an input of the shape (n_batches, n_losses)

            result = soft_min(batched_losses, alpha=self._alpha) ### error then HAS to be here....

            return result
        
        # Store strategies, loss functions, and residue pairs
        self._cyclization_loss = cyclization_loss
        self._loss_functions = loss_functions
        self._strategies_indices_pair_list = strategies_indices_pair_list
    
    def compute_loss(self, positions):
        """
        Compute the total cyclization loss with gradients enabled.

        Parameters:
        ----------
        positions (torch.Tensor): Tensor of shape (N_atoms, 3) representing the atomic positions.

        Returns:
        -------
        torch.Tensor: Total cyclization loss as a scalar.
        """
        with torch.enable_grad():  # Explicitly enable gradients for this computation
            return self._cyclization_loss(positions) # should be of shape (n_batch, )

    ### TODO: just need to make this method below batch friendly...    
    def get_smallest_loss(self, positions):
        """
        Identify the smallest cyclization loss, its type, and the amino acids involved for each batch.
        """
        # Compute batched losses for each loss function
        batched_losses = torch.stack([loss(positions) for loss in self._loss_functions], dim=1) 
        # Shape: (n_batch, n_loss_functions)

        # Find the index of the smallest loss for each batch
        min_indices = torch.argmin(batched_losses, dim=1)  # Shape: (n_batch,)

        # Get the strategy and indices for each batch
        results = [self._strategies_indices_pair_list[min_idx] for min_idx in min_indices] 
        # may be a way to speed ^^ this ^^ up, but not an issue since this only needs to be called once at the end of sampling.

        return results