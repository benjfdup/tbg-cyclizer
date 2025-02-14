### Imports ###
from abc import ABCMeta, abstractclassmethod, abstractmethod
from typing import Dict, final
from math import pi

import torch
import mdtraj as md

from pyclops.utils.utils import bond_angle_loss, dihedral_angle_loss, distance_loss, inherit_docstring
from pyclops.utils.constants import unit_scales
from pyclops.utils.IndexesMethodPair import IndexesMethodPair

######################################################################
# This class is a blueprint for all possible chemical losses.
# These losses are then considered in || by the loss handler.
######################################################################

# dihedral angle values -- use newman diagrams to visualizing.

class ChemicalLoss(metaclass=ABCMeta):
    '''
    A loss for organizing the individual chemical losses that are used in this project.
    '''

    # Class-level attribute for required keys; subclasses can override this
    indexes_keys = set()

    unit_scales = unit_scales
    
    def __init__(self,
                 units: str,
                 method: str, indexes: Dict[str, int], weights: Dict[str, float], 
                 offsets: Dict[str, float], use_bond_lengths: bool, use_bond_angles: bool, use_dihedrals: bool,
                 
                 bond_length_tolerance: float = None, bond_angle_tolerance: float = None, 
                 dihedral_tolerance: float = None,

                 device: torch.device = None
                 ):
        
        assert set(indexes.keys()) == self.indexes_keys, \
                f"The indexes dictionary must contain exactly the keys: {self.indexes_keys}. " \
                f"Received keys: {set(indexes.keys())}"
        
        self._units = units

        try:
            self._units_factor = self.unit_scales[units]
            # factor needed to convert atom positions to atom positions in angs.
        except KeyError as e:
            raise NotImplementedError('that unit is not implemented.')

        self._indexes = indexes
        self._weights = weights
        self._offsets = offsets

        assert method is not None, 'method must have a value'
        self._method = method

        self._use_bond_lengths = use_bond_lengths
        self._use_bond_angles = use_bond_angles
        self._use_dihedrals = use_dihedrals

        if use_bond_lengths:
            assert bond_length_tolerance is not None, 'bond_length_tolerance cannot be None if use_bond_lengths is not None'
        if use_bond_angles:
            assert bond_angle_tolerance is not None, 'bond_angle_tolerance cannot be None if use_bond_angles is not None'
            assert 0.0 <= bond_angle_tolerance < pi, 'bond_angle_tolerance must be in (0, pi]'
        if use_dihedrals:
            assert dihedral_tolerance is not None, 'dihedral_tolerance cannot be None if use_dihedrals is not None'
            assert 0.0 <= dihedral_tolerance < pi, 'dihedral_tolerance must be in (0, pi]'
        
        self._bond_length_tolerance = bond_length_tolerance
        self._bond_angle_tolerance = bond_angle_tolerance
        self._dihedral_tolerance = dihedral_tolerance

        self._device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # getters vvv
    @property
    def units(self) -> str:
        '''
        the units of the input atom positions.
        '''
        return self._units
    
    @property
    def units_factor(self) -> float:
        '''
        the factor to convert from atom position units to angstroms.
        
        for clarity: self.units_factor * positions = positions in angstroms
        '''
        return self._units_factor

    @property
    def indexes(self) -> Dict[str, int]:
        '''
        the indexes of the total atom positions used for the loss calculation.
        '''
        return self._indexes
    
    @property
    def weights(self) -> Dict[str, float]:
        '''
        the weights assigned to the bond length, bond angle and dihedral angle components of the chemical loss
        '''
        return self._weights
    
    @property
    def offsets(self) -> Dict[str, float]:
        '''
        the offsets assigned to the bond length, bond angle and dihedral angle components of the chemical loss
        '''
        return self._offsets
    
    @property
    def method(self) -> str:
        '''
        The method of cyclization that this loss represents.
        The method string should be ideally very descriptive, and should encode both the chemistry used
        and the indexes (and maybe even types) of the amino acids used for it.
        '''
        return self._method

    @property
    def use_bond_lengths(self) -> bool:
        '''
        whether or not the chemical loss should consider bond lengths.
        '''
        return self._use_bond_lengths

    @property
    def use_bond_angles(self) -> bool:
        '''
        whether or not the chemical loss should consider bond angles.
        '''
        return self._use_bond_angles

    @property
    def use_dihedrals(self) -> bool:
        '''
        whether or not the chemical loss should consider dihedral angles.
        '''
        return self._use_dihedrals
    
    @property
    def bond_length_tolerance(self) -> float:
        '''
        bond length tolerance (Å)
        '''
        return self._bond_length_tolerance
    
    @property
    def bond_angle_tolerance(self) -> float:
        '''
        bond angle tolerance (rads)
        '''
        return self._bond_angle_tolerance
    
    @property
    def dihedral_tolerance(self) -> float:
        '''
        dihedral tolerance (rads)
        '''
        return self._dihedral_tolerance
    
    @property
    def device(self) -> torch.device:
        return self._device
    # getters ^^^

    def calc_total_loss(self, 
                        distance_losses: torch.Tensor, # [n_batch, ]
                        bond_angle_losses: torch.Tensor, # [n_batch, ]
                        dihedral_losses: torch.Tensor, # [n_batch, ]
                        ):
        '''
        Calculates and returns the final total chemical loss.

        Args:
        ----
        distance_losses: torch.Tensor
            torch.Tensor of shape [n_batch] representing the distance losses of the positions
        bond_angle_losses: torch.Tensor
            torch.Tensor of shape [n_batch] representing the bond angle losses of the positions
        dihedral_losses: torch.Tensor
            torch.Tensor of shape [n_batch] representing the dihedral angle losses of the positions

        Returns:
        -------
        total_loss: torch.Tensor
            torch.Tensor of shape [n_batch] representing the summed total loss, with offsets and weights
            applied accordingly.
        '''

        total_loss = torch.zeros_like(distance_losses)

        if self.use_bond_lengths:
            total_loss += distance_losses * self.weights['bond_lengths'] + self.offsets['bond_lengths']
        
        if self.use_bond_angles:
            total_loss += bond_angle_losses * self.weights['bond_angles'] + self.offsets['bond_angles']
        
        if self.use_dihedrals:
            total_loss += dihedral_losses * self.weights['dihedral_angles'] + self.offsets['dihedral_angles']

        return total_loss

    @abstractclassmethod
    def get_indexes_and_methods(cls, traj: md.Trajectory, atom_indexes_dict: Dict) -> list:
        '''
        Returns a list of tuples, each of which denote an indexes, method str pair; this can then be used
        the losses __init__ method to initialize an instance of this subclass of ChemicalLoss

        Args:
        ----
        traj: md.Trajectory
            the mdtraj.Trajectory object of the pdb file that is being used for inference.

        atom_indexes_dict: Dict[(md.core.topology.Residue, str), int]
            a dict that maps from a residue object and an atom name (str) to an int, denoting that atom's index
            in the overall torch.Tensor. This can be easily derived from the above traj, but isn't in this function, 
            for speed and readability's sake.

        Returns:
        -------
        indexes_method_pairs_list: list[IndexesMethodPair]
            a list of IndexesMethodPair instances, each denoting an (atomic) indexes dict, method str pairing.
        '''
        pass
    
    @final
    def _convert_positions(self, positions: torch.Tensor) -> torch.Tensor:
        '''
        converts the input positions from their input units to angstroms
        '''

        return positions * self._units_factor
    
    @abstractmethod
    def _eval_loss(self, positions: torch.Tensor) -> torch.Tensor:
        '''
        Private method of ChemicalLoss objects, which does the actual computation of that
        particular chemical loss.

        Args:
        ----
        positions: torch.Tensor [batch_size, n_atoms, 3]
            tensor of atom positions, across the batch, in units of Angstroms.
        
        Out:
        ---
        losses: torch.Tensor [batch_size, ]
            the loss of each particular batch for this particular chemical loss.
        '''
        pass
    
    @final
    def __call__(self, positions: torch.Tensor):
        '''
        Evaluates the loss given all atoms positions of the chemically relevant atoms.
        Relevant atom indexes are stored at self.indexes.
        Positions should be of shape (batch_size, n_atoms, 3), but can be in the units specified
        by this particular chemical loss function
        '''

        true_pos = self._convert_positions(positions)
        loss = self._eval_loss(true_pos)

        return loss

######################################################################
# Below are the specific implementations of individual chemical losses
# for use in the cyclic loss handler.
######################################################################

class DisulfideLoss(ChemicalLoss):
    '''
    Loss of cyclization of disulfide bond between 2 cystines.
    '''

    indexes_keys = {'s1',
                    's2',
                    'b1',
                    'b2',
                    }

    def __init__(self, method: str, indexes: Dict[str, int], 
                 
                 # defaults below #
                 weights: Dict[str, float] = {'bond_lengths': 1, 'bond_angles': 1, 'dihedral_angles': 1},
                 offsets: Dict[str, float] = {'bond_lengths': 0, 'bond_angles': 0, 'dihedral_angles': 0},
                 use_bond_lengths: bool = True, 
                 use_bond_angles: bool = True, 
                 use_dihedrals: bool = True,
                 
                 bond_length_tolerance: float = 0.7, # get sources to support this
                 bond_angle_tolerance: float = 0.2,  # get sources to support this
                 dihedral_tolerance: float = 0.52, # best guess, 30 deg.

                 device: torch.device = None
                 ):
        
        '''
        Instantiates a disulfide loss object, which is a subclass of ChemicalLoss.

        Instantiates a disulfide loss object, whose job is largely to return the error associated with
        a Cysteine-Cysteine disulfide bridge for peptide cyclization. This is used to help compute a 
        generated sample's total cyclic loss. When called, this object returns the particular disulfide
        associated with a particular pair of Cysteine residues.

        Args:
        ----
        method: str
            the method of this particular disulfide loss. this string should be precise, as to encode that
            both a disulfide loss is being applied, and between which Cysteine residues this object is
            applied to.
        
        indexes: Dict[str, int]
            the atomic indexes that this particular loss fn accesses. Must contain keys 's1', 's2', 'b1', 'b2'

        ... finish docstring

        Returns:
        -------
        '''
        
        super().__init__(weights= weights, indexes= indexes, offsets= offsets, method= method, 
                         use_bond_lengths= use_bond_lengths, use_bond_angles= use_bond_angles, 
                         use_dihedrals= use_dihedrals,
                         
                         bond_length_tolerance= bond_length_tolerance, bond_angle_tolerance= bond_angle_tolerance, 
                         dihedral_tolerance= dihedral_tolerance,

                         device=device,
                         )

    def _eval_loss(self, positions: torch.Tensor) -> torch.Tensor:
        pos = positions # should be of shap (n_batch, n_atoms, 3)
        sulfur_index_1 = self._indexes['s1'] # sulfur of first cysteine
        sulfur_index_2 = self._indexes['s2'] # sulfur of second cysteine

        beta_index_1 = self._indexes['b1'] # beta carbon of first cysteine (side chain)
        beta_index_2 = self._indexes['b2'] # beta carbon of second cysteine (side chain)
        target_distance=2.05, # Typical bond length in Å
        
        target_bond_angle = torch.deg2rad(torch.tensor(105.6, device= self.device))
        # ^ https://doi.org/10.1038/npre.2011.6692.1 ^
        
        target_dihedral = torch.deg2rad(torch.tensor(90.0, device= self.device))
        # ^ https://pubs.rsc.org/en/content/articlepdf/2018/sc/c8sc01423j ^
        # to begin, I am only considering χ^3 from the above paper

        length_tolerance= self.bond_length_tolerance
        bond_angle_tolerance= self.bond_angle_tolerance
        dihedral_tolerance= self.dihedral_tolerance
        
        s1_atom = pos[:, sulfur_index_1, :].squeeze()
        s2_atom = pos[:, sulfur_index_2, :].squeeze()
        b1_atom = pos[:, beta_index_1, :].squeeze()
        b2_atom = pos[:, beta_index_2, :].squeeze()

        # Compute individual losses
        dist_loss = 0
        angle1_loss = 0
        angle2_loss = 0
        dihedral_loss = 0

        if self.use_bond_lengths: # verify bonding signs. How to do this?
            dist_loss += distance_loss(s1_atom, s2_atom, target_distance, length_tolerance)  # [batch_size, ]
        if self.use_bond_angles:
            angle1_loss += bond_angle_loss(b1_atom, s1_atom, s2_atom, target_bond_angle, bond_angle_tolerance)  # [batch_size, ]
            angle2_loss += bond_angle_loss(s1_atom, s2_atom, b2_atom, target_bond_angle, bond_angle_tolerance)  # [batch_size, ]
            # ^ probably want to normalize this somehow ^
        if self.use_dihedrals:
            dihedral_loss= torch.minimum(dihedral_angle_loss(b1_atom, s1_atom, s2_atom, b2_atom, target_dihedral, dihedral_tolerance), 
                                         dihedral_angle_loss(b1_atom, s1_atom, s2_atom, b2_atom, -1 * target_dihedral, dihedral_tolerance))
            # [batch_size, ] # consider dropping this out
    
        total_loss = self.calc_total_loss(distance_losses=dist_loss, 
                                          bond_angle_losses=angle1_loss + angle2_loss, 
                                          dihedral_losses= dihedral_loss)

        return total_loss #[n_batch, ]
    
    @inherit_docstring(ChemicalLoss.get_indexes_and_methods)
    @classmethod
    def get_indexes_and_methods(cls, traj: md.Trajectory, atom_indexes_dict: Dict) -> list:

        indexes_method_pairs_list = []

        residue_list = list(traj.topology.residues)
        cysteines = [r for r in residue_list if r.name == "CYS"]

        for i, cys_1 in enumerate(cysteines):
            for cys_2 in cysteines[i + 1:]:  # Avoid duplicate pairs
                sulfur_index_1 = atom_indexes_dict[(cys_1.index, 'SG')]
                sulfur_index_2 = atom_indexes_dict[(cys_2.index, 'SG')]
                beta_index_1 = atom_indexes_dict[(cys_1.index, "CB")]
                beta_index_2 = atom_indexes_dict[(cys_2.index, "CB")]

                indexes_dict= {
                    's1': sulfur_index_1,
                    's2': sulfur_index_2,
                    'b1': beta_index_1,
                    'b2': beta_index_2,
                }

                method_str = f'Disulfide, CYS {cys_1.index} -> CYS {cys_2.index}'

                indexes_method_pairs_list.append(IndexesMethodPair(indexes_dict, method_str))
    
        return indexes_method_pairs_list
    
class AmideAbstractLoss(ChemicalLoss, metaclass=ABCMeta):
    '''
    Abstract loss to handle all cyclic amide losses
    '''

    amide_bond_length=1.32 # Typical bond length in Å

######################################################################
# Below are the specific implementations of all Amine2CTerminal losses
# considered in this project. They all use the call defined below,
# but simply have differing get_indexes_and_methods implementations.
# Note that H2TAmide is somewhat arbitrarily included herein.
######################################################################

class Amine2CTerminal(AmideAbstractLoss, metaclass=ABCMeta): # TODO: put all amide commonalities here.
    '''
    Abstract loss to handle all amide losses where a (most often side-chain) amine is binding to a C terminal carbonyl.

    Included in this "family," somewhat arbitrarily, is the H2TAmideLoss
    '''
    indexes_keys = {'c', # carbonyl carbon
                    'o', # carnonyl oxygen
                    'ca', # alpha carbon of amino acid with the BINDING carbonyl group.
                    'n', # amide nitrogen
                    'h1', # amide hydrogen 1
                    'h2', # amide hydrogen 2
                    }

    def _eval_loss(self, positions: torch.Tensor) -> torch.Tensor:
        pos = positions # should be of shape (n_batch, n_atoms, 3)
        carbonyl_carbon_index = self._indexes['c'] # carbonyl carbon
        oxygen_index = self._indexes['o'] # carbonyl oxygen
        carbon_alpha_index = self._indexes['ca'] # carbon alpha
        amide_nitrogen_index = self._indexes['n'] # amide nitrogen
        amide_hydrogen_1_index = self._indexes['h1'] # amide hydrogen 1
        amide_hydrogen_2_index = self._indexes['h2'] # amide hydrogen 2

        target_distance= self.amide_bond_length, # Typical bond length in Å
        #^ https://doi.org/10.1016/B978-0-12-095461-2.00004-7 ^#
        target_ca_cyl_n_angle = torch.deg2rad(torch.tensor(114.0, device= self.device))
        target_peptide_dihedral_angle = torch.deg2rad(torch.tensor(0.0, device= self.device)) # peptide bond components are planar
        #^ https://doi.org/10.1016/B978-0-12-095461-2.00004-7 ^#

        cyl_atom = pos[:, carbonyl_carbon_index, :].squeeze()
        oxy_atom = pos[:, oxygen_index, :].squeeze()
        ca_atom = pos[:, carbon_alpha_index, :].squeeze()
        n_atom = pos[:, amide_nitrogen_index, :].squeeze()
        h1_atom = pos[:, amide_hydrogen_1_index, :].squeeze()
        h2_atom = pos[:, amide_hydrogen_2_index, :].squeeze()

        length_tolerance= self.bond_length_tolerance
        bond_angle_tolerance= self.bond_angle_tolerance
        dihedral_tolerance= self.dihedral_tolerance

        # Compute individual losses
        dist_loss = 0
        angle1_loss = 0
        dihedral_loss = 0

        if self.use_bond_lengths: # verify bonding signs. How to do this?
            dist_loss += distance_loss(cyl_atom, n_atom, target_distance, length_tolerance)  # [batch_size, ]
        if self.use_bond_angles:
            angle1_loss += bond_angle_loss(ca_atom, cyl_atom, n_atom, target_ca_cyl_n_angle, bond_angle_tolerance)  # [batch_size, ]
        if self.use_dihedrals:
            dihedral_loss= torch.minimum(dihedral_angle_loss(oxy_atom, cyl_atom, n_atom, h1_atom, 
                                                             target_peptide_dihedral_angle, dihedral_tolerance), 
                                         dihedral_angle_loss(oxy_atom, cyl_atom, n_atom, h2_atom, 
                                                             target_peptide_dihedral_angle, dihedral_tolerance))
            # C, O, N, H atom lie all in the same plane.
            # [batch_size, ]
    
        total_loss = self.calc_total_loss(distance_losses=dist_loss, 
                                          bond_angle_losses=angle1_loss, 
                                          dihedral_losses= dihedral_loss)
        
        return total_loss
    
class H2TAmideLoss(Amine2CTerminal):
    '''
    Loss of cyclization of a head to tail amide bond
    '''
    
    @inherit_docstring(ChemicalLoss.get_indexes_and_methods)
    @classmethod
    def get_indexes_and_methods(cls, traj: md.Trajectory, atom_indexes_dict: Dict) -> list:

        indexes_method_pairs_list = []
        
        residue_list = list(traj.topology.residues)
        if len(residue_list) < 2:
            return indexes_method_pairs_list  # Not enough residues for a cyclic bond

        first_res = residue_list[0]  # N-terminal
        last_res = residue_list[-1]  # C-terminal

        # Get relevant atom indexes
        indexes_dict = {
            'c': atom_indexes_dict[(last_res.index, 'C')],   # Carbonyl carbon from the last residue
            'o': atom_indexes_dict[(last_res.index, 'O')],   # Carbonyl oxygen from the last residue
            'ca': atom_indexes_dict[(last_res.index, 'CA')], # Alpha carbon of last residue
            'n': atom_indexes_dict[(first_res.index, 'N')],  # Amide nitrogen from the first residue
            'h1': atom_indexes_dict[(first_res.index, 'H1')],  # Amide hydrogen 1
            'h2': atom_indexes_dict[(first_res.index, 'H2')],  # Amide hydrogen 2
            }

        method_str = f'Head-to-Tail Amide Bond, {first_res.name} {first_res.index} -> {last_res.name} {last_res.index}'

        indexes_method_pairs_list.append(IndexesMethodPair(indexes_dict, method_str))

        return indexes_method_pairs_list

class Lysine2CTerminalLoss(AmideAbstractLoss): # try to find strategies in the literature before including them here...
    '''
    Loss of cyclization of a Lysine sidechain amine group to a peptide C-terminal.
    This handles cases where a peptide is cyclized via the ε-amine group of lysine 
    bonding to the C-terminal carbonyl of the peptide.
    '''

    @inherit_docstring(ChemicalLoss.get_indexes_and_methods)
    @classmethod
    def get_indexes_and_methods(cls, traj: md.Trajectory, 
                                atom_indexes_dict: Dict) -> list:

        indexes_method_pairs_list = []
        
        residue_list = list(traj.topology.residues)
        if len(residue_list) < 2:
            return indexes_method_pairs_list  # Not enough residues for a cyclic bond

        last_res = residue_list[-1]  # C-terminal residue

        # Find all lysine residues (LYS)
        lysines = [r for r in residue_list if r.name == "LYS"]

        for lys in lysines:
            # Ensure necessary atoms exist in the mapping
            if (lys.index, 'NZ') not in atom_indexes_dict or \
               (lys.index, 'H1') not in atom_indexes_dict or \
               (lys.index, 'H2') not in atom_indexes_dict or \
               (last_res.index, 'C') not in atom_indexes_dict or \
               (last_res.index, 'O') not in atom_indexes_dict or \
               (last_res.index, 'CA') not in atom_indexes_dict:
                #continue  # Skip if any required atoms are missing
                raise NotImplementedError('Lysine is missing atoms...')

            # Define atomic index mapping
            indexes_dict = {
                'c': atom_indexes_dict[(last_res.index, 'C')],   # Carbonyl carbon from last residue
                'o': atom_indexes_dict[(last_res.index, 'O')],   # Carbonyl oxygen from last residue
                'ca': atom_indexes_dict[(last_res.index, 'CA')], # Alpha carbon of last residue
                'n': atom_indexes_dict[(lys.index, 'NZ')],       # Amide nitrogen from Lysine sidechain
                'h1': atom_indexes_dict[(lys.index, 'H1')],      # Amide hydrogen 1 (assumed)
                'h2': atom_indexes_dict[(lys.index, 'H2')],      # Amide hydrogen 2 (assumed)
            }

            # Create descriptive method string
            method_str = f'Lysine ε-Amine to C-Terminal Bond, LYS {lys.index} -> {last_res.name} {last_res.index}'

            # Append result
            indexes_method_pairs_list.append(IndexesMethodPair(indexes_dict, method_str))

        return indexes_method_pairs_list

######################################################################
# Above are the specific implementations of all Amine2CTerminal losses
# considered in this project. They all use the call defined below,
# but simply have differing get_indexes_and_methods implementations.
# Note that H2TAmide is somewhat arbitrarily included herein.
######################################################################