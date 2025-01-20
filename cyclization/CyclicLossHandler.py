from typing import Type, Set, Dict

import torch
import mdtraj as md

import ChemicalLoss as cl
from utils import soft_min

class CyclicLossHandler():

    bonding_atoms = {"SG", "CB", "C", "CA", "N", "H", "NZ", "CE", "CG", "OG", "SD", "NE", "CD", }
    
    def __init__(self, pdb_path: str, 
                # add weights? add offsets? add use_bond_lengths, add use_bond_angles,
                # add use_dihedrals?
                 strategies: Set[Type[cl.ChemicalLoss]] = {cl.DisulfideLoss, },
                 weights: Dict[str, float] = {'bond_lengths': 1.0, 'bond_angles': 1.0, 'dihedral_angles': 1.0}, 
                 offsets: Dict[str, float] = {'bond_lengths': 0.0, 'bond_angles': 0.0, 'dihedral_angles': 0.0},
                 use_bond_lengths: bool = True, use_bond_angles: bool = True, use_dihedrals: bool = True,
                 alpha: float= -3.0,
                 ):
        
        self._pdb_path = pdb_path
        self._strategies = strategies
        self._alpha = alpha
        self._losses = self._initialize_losses()
        self._weights = weights
        self._offsets = offsets

        self._use_bond_lengths = use_bond_lengths
        self._use_bond_angles = use_bond_angles
        self._use_dihedrals = use_dihedrals
        
    # getters vvv
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
    def losses(self) -> list[cl.ChemicalLoss]:
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

                loss = strat.__init__(method=method_str, indexes=indexes, 
                                      weights= self.weights, offsets= self.offsets,
                                      use_bond_lengths= self.use_bond_lengths,
                                      use_bond_angles= self.use_bond_angles,
                                      use_dihedrals= self.use_dihedrals,
                                      )
                
                losses.append(loss)

        return losses


    def __call__(self, positions: torch.Tensor) -> torch.Tensor:
        batched_losses = torch.stack([loss(positions) for loss in self.losses], dim=1).squeeze() # [n_batches, n_losses]
        
        return soft_min(batched_losses) # [n_batches, ]