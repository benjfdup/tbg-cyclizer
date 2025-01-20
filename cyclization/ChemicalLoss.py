### Imports ###
from abc import ABC, abstractmethod
from typing import Dict
from math import pi

import torch
import mdtraj as md

from utils import bond_angle_loss, dihedral_angle_loss, distance_loss, inherit_docstring
from IndexesMethodPair import IndexesMethodPair

######################################################################
# This class is a blueprint for all possible chemical losses.
# These losses are then considered in || by the loss handler.
######################################################################

class ChemicalLoss(ABC):
    '''
    A loss for organizing the individual chemical losses that are used in this project.
    '''
    
    def __init__(self, method: str, indexes: Dict[str, int], weights: Dict[str, float], 
                 offsets: Dict[str, float], use_bond_lengths: bool, use_bond_angles: bool, use_dihedrals: bool,
                 
                 bond_length_tolerance: float = None, bond_angle_tolerance: float = None, 
                 dihedral_tolerance: float = None,

                 device: torch.device = None
                 ):
        
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
        return self.device
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
    
    @abstractmethod
    def __call__(self, positions: torch.Tensor):
        '''
        Evaluates the loss given all atoms positions of the chemically relevant atoms.
        Relevant atom indexes are stored at self.indexes.
        Positions should be of shape (batch_size, n_atoms, 3).
        '''
        pass

    @abstractmethod
    @classmethod
    def get_indexes_and_methods(cls, traj: md.Trajectory, atom_indexes_dict: Dict[(int, str), int]) -> list[IndexesMethodPair]:
        '''
        Returns a list of tuples, each of which denote an indexes, method str pair; this can then be used
        the losses __init__ method to initialize an instance of it

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

######################################################################
# Below are the specific implementations of individual chemical losses
# for use in the cyclic loss handler.
######################################################################

class DisulfideLoss(ChemicalLoss):
    '''
    Loss of cyclization of disulfide bond between 2 cystines.
    '''

    indexes_keys = {'s1', 's2', 'b1', 'b2',}

    def __init__(self, method: str, indexes: Dict[str, int], 
                 
                 # defaults below #
                 weights: Dict[str, float] = {'bond_lengths': 1, 'bond_angles': 1, 'dihedral_angles': 1},
                 offsets: Dict[str, float] = {'bond_lengths': 0, 'bond_angles': 0, 'dihedral_angles': 0},
                 use_bond_lengths: bool = True, use_bond_angles: bool = True, use_dihedrals: bool = True,
                 bond_length_tolerance: float = 0.2, bond_angle_tolerance: float = 0.1, dihedral_tolerance: float = 0.1,

                 device: torch.device = None
                 ):
        
        '''
        indexes must contain keys 's1', 's2', 'b1', 'b2'
        '''

        # Assert that the keys in indexes are exactly the required keys
        assert set(indexes.keys()) == DisulfideLoss.indexes_keys, \
            f"The indexes dictionary must contain exactly the keys: {DisulfideLoss.indexes_keys}. " \
            f"Received keys: {set(indexes.keys())}"
        
        super().__init__(weights= weights, indexes= indexes, offsets= offsets, method= method, 
                         use_bond_lengths= use_bond_lengths, use_bond_angles= use_bond_angles, 
                         use_dihedrals= use_dihedrals,
                         
                         bond_length_tolerance= bond_length_tolerance, bond_angle_tolerance= bond_angle_tolerance, 
                         dihedral_tolerance= dihedral_tolerance,

                         device=device ,
                         )

    def __call__(self, positions: torch.Tensor) -> torch.Tensor:
        pos = positions # should be of shap (n_batch, n_atoms, 3)
        sulfur_index_1 = self._indexes['s1']
        sulfur_index_2 = self._indexes['s2']
        beta_index_1 = self._indexes['b1']
        beta_index_2 = self._indexes['b2']
        target_distance=2.05, # Typical bond length in Å (?)
        
        target_bond_angle = torch.deg2rad(torch.tensor(102.5, device= self.device))
        target_dihedral = torch.deg2rad(torch.tensor(90.0, device= self.device))

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

        if self.use_bond_lengths:
            dist_loss += distance_loss(s1_atom, s2_atom, target_distance, length_tolerance)  # Shape: (batch_size)
        if self.use_bond_angles:
            angle1_loss += bond_angle_loss(b1_atom, s1_atom, s2_atom, target_bond_angle, bond_angle_tolerance)  # Shape: (batch_size)
            angle2_loss += bond_angle_loss(s1_atom, s2_atom, b2_atom, target_bond_angle, bond_angle_tolerance)  # Shape: (batch_size)
        if self.use_dihedrals:
            dihedral_loss = dihedral_angle_loss(b1_atom, s1_atom, s2_atom, b2_atom, target_dihedral, dihedral_tolerance)  # Shape: (batch_size)
    
        total_loss = self.calc_total_loss(distance_losses=dist_loss, 
                                          bond_angle_losses=angle1_loss + angle2_loss, 
                                          dihedral_losses= dihedral_loss)

        return total_loss # should be of shape [n_batch, ]
    
    @inherit_docstring(ChemicalLoss.get_indexes_and_methods)
    @classmethod
    def get_indexes_and_methods(cls, traj: md.Trajectory, atom_indexes_dict: Dict[(int, str), int]) -> list[IndexesMethodPair]:

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

                method_str = f'Disulfide, CYS {cys_1.index} & CYS {cys_2.index}'

                indexes_method_pairs_list.append(IndexesMethodPair(indexes_dict, method_str))
        
        return indexes_method_pairs_list
    
######################################################################
# What is left to do:
# Implement the loss strategies discussed by Alex.
# Particularly, I need the distances between 'cannonical' atoms.
# the bond angles formed by cannonical atoms.
# and the dihedral angles formed by cannonical atoms.
######################################################################