from abc import ABC, abstractmethod
from typing import Type, Set, Dict

import torch
import mdtraj as md

import pyclops.losses.ChemicalLoss as cl
from pyclops.utils.utils import soft_min
from pyclops.utils.constants import unit_scales
from pyclops.losses.LossCoeff import LossCoeff

class LossHandler(ABC):
    @abstractmethod
    def __call__(self, positions: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        '''
        A method which calculates a loss for the atom positions in a given generated peptide.

        Note, positions should be be of shape [n_batch, n_atoms, 3]
        '''
        pass

class CyclicLossHandler(LossHandler):
    bonding_atoms = {"SG", "CB", "C", "CA", "N", "H", "NZ", "CE", "CG", "OG", "SD", "NE", "CD", }
    
    def __init__(self,
                 units: str, 
                 pdb_path: str, 
                # add weights? add offsets? add use_bond_lengths, add use_bond_angles,
                # add use_dihedrals?
                 strategies: Set[Type[cl.ChemicalLoss]] = {cl.DisulfideLoss, },
                 weights: Dict[str, float] = {'bond_lengths': 1.0, 'bond_angles': 1.0, 'dihedral_angles': 1.0}, 
                 offsets: Dict[str, float] = {'bond_lengths': 0.0, 'bond_angles': 0.0, 'dihedral_angles': 0.0},
                 use_bond_lengths: bool = True, use_bond_angles: bool = True, use_dihedrals: bool = True,
                 alpha: float= -3.0,
                 ):
        
        self._units = units
        self._pdb_path = pdb_path
        self._strategies = strategies
        self._alpha = alpha
        self._weights = weights
        self._offsets = offsets

        self._use_bond_lengths = use_bond_lengths
        self._use_bond_angles = use_bond_angles
        self._use_dihedrals = use_dihedrals

        self._losses = self._initialize_losses()
        
    # getters vvv
    @property
    def units(self) -> str:
        '''
        should only be used to be passed to the relevant chemical losses.
        '''
        return self._units

    @property
    def pdb_path(self) -> str:
        return self._pdb_path
    
    @property
    def strategies(self) -> Set[Type[cl.ChemicalLoss]]:
        return self._strategies
    
    @property
    def alpha(self) -> float:
        return self._alpha
    
    @property
    def losses(self) -> list:
        return self._losses
    
    @property
    def weights(self) -> Dict[str, float]:
        return self._weights
    
    @property
    def offsets(self) -> Dict[str, float]:
        return self._offsets
    
    @property
    def use_bond_lengths(self) -> bool:
        return self._use_bond_lengths
    
    @property
    def use_bond_angles(self) -> bool:
        return self._use_bond_angles
    
    @property
    def use_dihedrals(self) -> bool:
        return self._use_dihedrals

    @property
    def traj(self) -> md.Trajectory:
        return md.load(self._pdb_path)
    
    @property
    def topology(self) -> md.Trajectory.topology:
        traj = self.traj
        return traj.topology
    # getters ^^^

    @staticmethod
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
    
    def _initialize_losses(self):
        traj = self.traj
        atom_indexes_dict = CyclicLossHandler.precompute_atom_indices(list(self.topology.residues), 
                                                                      CyclicLossHandler.bonding_atoms) 
        # not optimal, but only run once, so I dont think its so worth redoing all of this.
        losses = []
        
        for strat in self.strategies:
            indexes_methods_list = strat.get_indexes_and_methods(traj, atom_indexes_dict)

            for pair in indexes_methods_list:
                indexes = pair.indexes
                method_str = pair.method

                loss = strat(
                             units= self.units,
                             method=method_str, indexes=indexes,
                             weights= self.weights, offsets= self.offsets,
                             use_bond_lengths= self.use_bond_lengths,
                             use_bond_angles= self.use_bond_angles,
                             use_dihedrals= self.use_dihedrals,
                             )
                
                losses.append(loss)

        return losses

    def __call__(self, positions: torch.Tensor) -> torch.Tensor:
        '''
        Args:
        ----
        positions: torch.Tensor [batch_size, n_atoms, 3]
            a torch.Tensor denoting atom positions cross batch
        
        Returns:
        -------
        loss: torch.Tensor [batch_size, ]
            the cyclic loss of each batch
        '''
        batched_losses = torch.stack([loss(positions) for loss in self.losses], dim=1).squeeze() # [n_batches, n_losses]
        loss = soft_min(batched_losses) # [n_batches, ]
        
        return loss
    
    def get_smallest_loss(self, positions: torch.Tensor) -> list:
        batched_losses = torch.stack([loss(positions) for loss in self.losses], dim=1).squeeze() # [n_batches, n_losses]

        min_indexes = torch.argmin(batched_losses, dim= 1) # [n_batches, ]

        # Use the indices to cobble together the corresponding loss objects
        smallest_losses = [self.losses[idx] for idx in min_indexes.tolist()]
        
        return smallest_losses # len= n_batches
    
    def get_smallest_loss_methods(self, positions: torch.Tensor) -> list:
        loss_list = self.get_smallest_loss(positions)
        smallest_loss_methods = [loss.method for loss in loss_list]

        return smallest_loss_methods
        
    def eval_smallest_loss(self, positions: torch.Tensor) -> torch.Tensor:
        batched_losses = torch.stack([loss(positions) for loss in self.losses], dim=1).squeeze() # [n_batches, n_losses]

        return torch.min(batched_losses, dim= 1)

class GyrationLossHandler(LossHandler):
    def __init__(self, units: str, 
                 squared: bool = False):
        self._squared = squared
        self._units = units

        try:
            self._units_factor = unit_scales[units]
        except KeyError as e:
            raise NotImplementedError('that unit is not implemented.')
    
    @property
    def units(self) -> str:
        return self._units
    
    @property
    def units_factor(self) -> float:
        return self._units_factor
    
    @property
    def squared(self) -> bool:
        return self._squared

    def __call__(self, positions: torch.Tensor) -> torch.Tensor:
        
        # pos: [n_batch, n_atoms, 3]
        pos = positions * self.units_factor # converts to the correct units.
        
        # Step 1: Compute the center of mass (mean position) for each batch
        center_of_mass = torch.mean(pos, dim=1, keepdim=True)  # [n_batch, 1, 3]
        
        # Step 2: Compute the squared distances from each atom to the center of mass
        squared_distances = torch.sum((pos - center_of_mass) ** 2, dim=-1)  # [n_batch, n_atoms]
        
        # Step 3: Compute the mean squared distance for each batch
        mean_squared_distance = torch.mean(squared_distances, dim=-1)  # [n_batch, ]
        
        # Step 4: Return either the squared radius of gyration or its square root
        if self.squared:
            val = mean_squared_distance  # Return R_g^2
        else:
            val = torch.sqrt(mean_squared_distance)  # Return R_g
        
        return val # [n_batch, ]
        
class GyrationCyclicLossHandler(LossHandler):
    def __init__(self, l_cyclic: CyclicLossHandler, l_gyr: GyrationLossHandler, gamma: LossCoeff):
        self.gamma = gamma
        self.l_cyclic = l_cyclic
        self.l_gyr = l_gyr
    
    def get_smallest_loss(self, positions: torch.Tensor) -> list:
        return self.l_cyclic.get_smallest_loss(positions)
    
    def get_smallest_loss_methods(self, positions: torch.Tensor) -> list:
        return self.l_cyclic.get_smallest_loss_methods(positions)
        
    def eval_smallest_loss(self, positions: torch.Tensor) -> torch.Tensor:
        return self.l_cyclic.eval_smallest_loss(positions)
    
    def __call__(self, positions: torch.Tensor, t: torch.Tensor):
        g_t = self.gamma(t) # [batch_size, ]

        l_gyr = self.l_gyr(positions = positions) # [batch_size, ]
        l_cyc = self.l_cyclic(positions = positions) # [batch_size, ]

        return g_t * l_gyr + (1 - g_t) * l_cyc