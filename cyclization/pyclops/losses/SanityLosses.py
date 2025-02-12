import torch

from ChemicalLoss import AmideAbstractLoss, ChemicalLoss
from pyclops.utils.utils import bond_angle_loss, dihedral_angle_loss, distance_loss, inherit_docstring
from pyclops.utils.IndexesMethodPair import IndexesMethodPair
import mdtraj as md

# the below losses are exclusively for the l1 sanity check.
class ProC2PheN(AmideAbstractLoss): #true cyclization strategy used.
    '''
    Loss to handle just the distance between the Proline C terminal and Phenylalanine N terminal.
    '''

    indexes_keys = {'c', # proline carbonyl carbon
                    'phe_n', # phenylalanine N terminal nitrogen
                    }
    
    def __call__(self, positions: torch.Tensor) -> torch.Tensor:
        pos = positions

        proline_c_index = self._indexes['c']
        phen_n_index = self._indexes['phe_n']

        target_distance = self.amide_bond_length

        length_tolerance= self.bond_length_tolerance

        proline_c_atom = pos[:, proline_c_index, :].squeeze()
        phen_n_atom = pos[:, phen_n_index, :].squeeze()

        dist_loss = torch.zeros(pos.shape[0])
        angle_loss = torch.zeros(pos.shape[0])
        dihedral_loss = torch.zeros(pos.shape[0])

        if self.use_bond_lengths: # verify bonding signs. How to do this?
            dist_loss += distance_loss(proline_c_atom, phen_n_atom, target_distance, length_tolerance)  # [batch_size, ]
        
        tot_loss = self.calc_total_loss(dist_loss, angle_loss, dihedral_loss)
        
        return tot_loss #[n_batch, ]
    
    @inherit_docstring(ChemicalLoss.get_indexes_and_methods)
    @classmethod
    def get_indexes_and_methods(cls, traj: md.Trajectory, atom_indexes_dict: dict) -> list:

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
    


    ### TEMPLATE ###
    @inherit_docstring(ChemicalLoss.get_indexes_and_methods)
    @classmethod
    def get_indexes_and_methods(cls, traj: md.Trajectory, atom_indexes_dict: dict) -> list:

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