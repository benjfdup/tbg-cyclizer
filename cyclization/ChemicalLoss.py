### Imports ###
from abc import ABC, abstractmethod
from typing import Dict
from math import pi

import torch

from utils import bond_angle_loss, dihedral_angle_loss, distance_loss

######################################################################
# This class is a blueprint for all possible chemical losses.
# These losses are then considered in || by the loss handler.
######################################################################

class ChemicalLoss(ABC):
    '''
    A loss for organizing the individual chemical losses that are used in this project.
    '''

    def __init__(self,  method: str, indexes: list[int], weights: Dict[str, float], 
                 offsets: Dict[str, float], use_bond_lengths: bool, use_bond_angles: bool, use_dihedrals: bool,
                 
                 bond_length_tolerance: float = None, bond_angle_tolerance: float = None, 
                 dihedral_tolerance: float = None,
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

    # getters vvv
    @property
    def indexes(self) -> list[int]:
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
        return self._indexes

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
    # getters ^^^

    def calc_total_loss(self, distance_losses: torch.tensor, bond_angle_losses: torch.tensor, dihedral_losses: torch.tensor):
        '''
        calculates and returns the final total chemical loss.
        '''
        total_loss = torch.zeros_like(distance_losses)

        if self.use_bond_lengths:
            total_loss += distance_losses * self.weights['bond_lenghts'] + self.offsets['bond_lengths']
        
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

######################################################################
# Below are the specific implementations of individual chemical losses
# for use in the cyclic loss handler.
######################################################################

class DisulfideLoss(ChemicalLoss):
    '''
    Loss of cyclization of disulfide bond between 2 cystines.
    '''
    def __init__(self, method: str, indexes: list[int], 
                 
                 # defaults below #
                 weights: Dict[str, float] = {'bond_lengths': 1, 'bond_angles': 1, 'dihedral_angles': 1},
                 offsets: Dict[str, float] = {'bond_lengths': 0, 'bond_angles': 0, 'dihedral_angles': 0},
                 use_bond_lengths: bool = True, use_bond_angles: bool = True, use_dihedrals: bool = True,
                 bond_length_tolerance: float = 0.2, bond_angle_tolerance: float = 0.1, dihedral_tolerance: float = 0.1,
                 ):
        
        super().__init__(weights= weights, indexes= indexes, offsets= offsets, method= method, 
                         use_bond_lengths= use_bond_lengths, use_bond_angles= use_bond_angles, 
                         use_dihedrals= use_dihedrals,
                         
                         bond_length_tolerance= bond_length_tolerance, bond_angle_tolerance= bond_angle_tolerance, 
                         dihedral_tolerance= dihedral_tolerance,
                         )

    def __call__(self, positions: torch.Tensor):
        pos = positions
        sulfur_index_1, sulfur_index_2, beta_index_1, beta_index_2 = self._indexes
        target_distance=2.05, # Typical bond length in Å (?)
        target_bond_angle = torch.deg2rad(torch.tensor(102.5))
        target_dihedral = torch.deg2rad(torch.tensor(90.0))

        # may want to make these attributes of the outer loss object.
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

        return total_loss
    
######################################################################
# What is left to do:
# Implement the loss strategies discussed by Alex.
# Particularly, I need the distances between 'cannonical' atoms.
# the bond angles formed by cannonical atoms.
# and the dihedral angles formed by cannonical atoms.
######################################################################