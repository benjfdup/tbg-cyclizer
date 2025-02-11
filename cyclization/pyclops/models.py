import importlib
import sys

import torch

# Path to tbg package
tbg_spec = importlib.util.spec_from_file_location(
    "tbg", "/home/bfd21/rds/hpc-work/tbg/tbg/__init__.py"
)
tbg = importlib.util.module_from_spec(tbg_spec)
tbg_spec.loader.exec_module(tbg)

# Manually register the tbg package so that normal imports work
sys.modules["tbg"] = tbg

from tbg.models2 import EGNN
from tbg.utils import remove_mean

from pyclops.utils.utils import generate_bb_all_sc_adjacent_from_pdb
from pyclops.losses.LossCoeff import LossCoeff
from pyclops.losses.LossHandler import GyrationCyclicLossHandler, CyclicLossHandler, GyrationLossHandler

######################################################################
# vv Training Classes vv
# Classes with custom graph pruning which are good to do training on
######################################################################

class EGNN_dynamics_AD2_cat(torch.nn.Module):
    def __init__(self, n_particles, n_dimension,h_initial, hidden_nf=64, device='cpu',
            act_fn=torch.nn.SiLU(), n_layers=4, recurrent=True, attention=False,
                 condition_time=True, tanh=False, mode='egnn_dynamics', agg='sum'):
        super().__init__()
        self.mode = mode
        # Initial one hot encoding of the different element types
        self.h_initial = h_initial

        if mode == 'egnn_dynamics':
            h_size = h_initial.size(1)
            if condition_time:
                h_size += 1
            
            self.egnn = EGNN(in_node_nf=h_size, 
                             in_edge_nf=1, 
                             hidden_nf=hidden_nf, 
                             device=device, 
                             act_fn=act_fn, 
                             n_layers=n_layers, 
                             recurrent=recurrent, 
                             attention=attention, 
                             tanh=tanh, 
                             agg=agg,
                             )
        else:
            raise NotImplemented()

        self.device = device
        self._n_particles = n_particles
        self._n_dimension = n_dimension
        self.edges = self._create_edges()
        self._edges_dict = {}
        self.condition_time = condition_time
        # Count function calls
        self.counter = 0
        
    # confused on how t works... but ok?
    def forward(self, t, xs):

        n_batch = xs.shape[0]
        edges = self._cast_edges2batch(self.edges, n_batch, self._n_particles)
        edges = [edges[0], edges[1]]
        #Changed by Leon
        x = xs.reshape(n_batch*self._n_particles, self._n_dimension).clone()
        h = self.h_initial.to(self.device).reshape(1,-1)
        h = h.repeat(n_batch, 1)
        h = h.reshape(n_batch*self._n_particles, -1)
        # node compatability
        # print(t.shape)
        t = torch.tensor(t).to(xs)
        if t.shape != (n_batch,1):
            t = t.repeat(n_batch)
        t = t.repeat(1, self._n_particles)
        t = t.reshape(n_batch*self._n_particles, 1)
        #print(t.shape, h.shape)
        #print(t)
        if self.condition_time:
            h = torch.cat([h, t], dim=-1)
        else:
            h = h.float()
        if self.mode == 'egnn_dynamics':
            edge_attr = torch.sum((x[edges[0]] - x[edges[1]])**2, dim=1, keepdim=True)
            _, x_final = self.egnn(h, x, edges, edge_attr=edge_attr)
            vel = x_final - x

        else:
            raise NotImplemented()
            
        vel = vel.view(n_batch, self._n_particles, self._n_dimension)
        vel = remove_mean(vel)
        #print(t, xs)
        self.counter += 1
        return vel.view(n_batch,  self._n_particles* self._n_dimension)

    def _create_edges(self):
        rows, cols = [], []
        for i in range(self._n_particles):
            for j in range(i + 1, self._n_particles):
                rows.append(i)
                cols.append(j)
                rows.append(j)
                cols.append(i)
        return [torch.LongTensor(rows), torch.LongTensor(cols)]

    def _cast_edges2batch(self, edges, n_batch, n_nodes):
        if n_batch not in self._edges_dict:
            self._edges_dict = {}
            rows, cols = edges
            rows_total, cols_total = [], []
            for i in range(n_batch):
                rows_total.append(rows + i * n_nodes)
                cols_total.append(cols + i * n_nodes)
            rows_total = torch.cat(rows_total).to(self.device)
            cols_total = torch.cat(cols_total).to(self.device)

            self._edges_dict[n_batch] = [rows_total, cols_total]
        return self._edges_dict[n_batch]

class EGNN_dynamics_AD2_cat_bb_all_sc_adjacent(EGNN_dynamics_AD2_cat): # conditioned on time
    def __init__(self, *args, pdb_file: str= None, device: torch.device= None, condition_time: bool = True, **kwargs):
        self.counter = 0
        self._custom_adj_matrix = None
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Remove subclass-specific arguments from kwargs before passing to super()
        pdb_file = kwargs.pop('pdb_file', pdb_file)
        condition_time = kwargs.pop('condition_time', condition_time)

        # Call the superclass constructor with filtered kwargs
        super().__init__(*args, device=self.device, condition_time=condition_time, **kwargs)

        # Handle subclass-specific logic
        self.pdb_file = pdb_file

        if pdb_file:
            # Generate a custom adjacency matrix from the PDB file
            edge_tensor = generate_bb_all_sc_adjacent_from_pdb(pdb_file)
            self._custom_adj_matrix = edge_tensor
            self.edges = edge_tensor.to(self.device)
        else:
            raise NotImplemented()

        # Initialize self.edges after the subclass attributes are set
        #self.edges = self._create_edges()

    def _create_edges(self):
        """
        Override the edge creation logic to return the custom adjacency matrix.
        """
        if self._custom_adj_matrix is not None:
            # The custom adjacency matrix already has the correct format
            return self._custom_adj_matrix
        else:
            # Fallback to the default behavior if no custom adjacency is provided
            return super()._create_edges()
        
######################################################################
# ^^ Training Classes ^^
# Classes with custom graph pruning which are good to do training on.
######################################################################

######################################################################
# vv Sampling Classes vv Bad for training.
# First bunch is conditioned on time.
# latter bunch is unconditioned on time.
######################################################################
    
class EGNN_dynamics_AD2_cat_bb_all_sc_adj_cyclic(EGNN_dynamics_AD2_cat_bb_all_sc_adjacent): # likely need to clean this code up...
    # just cyclic, conditioned on time.
    '''
    This model should not be used in training.

    A class representing the NetDynamics of all backbone atoms fully connected to one another, but with sidechain atoms only 
    connected fully to other atoms within their, or adjacent, amino acids.

    Parameters:
    ----------
    l_cyclic (callable: torch.Tensor -> torch.Tensor): 
        a callable which takes a tensor of shape (n_batch, n_atoms, n_dimensions), representing the atom positions, 
        at that point in time, accross batched, and which outputs a single number [torch.Tensor, shape (n_batch), )], 
        representing the loss associated with cyclicality. Must be a positive number for all batches. All computation 
        for this should be done on the gpu.

    w_t (callable: float -> float):
        a callable of a float between 0 & 1 representing time, which outputs another float between 0 & 1, 
        representing the loss coefficient [torch.Tensor, shape (n_batch, )].
    '''

    def __init__(self, *args, w_t: LossCoeff, l_cyclic: CyclicLossHandler, 
                 condition_time: bool = True, with_dlogp: bool=True, **kwargs):
        super().__init__(*args, condition_time = condition_time, **kwargs)
        
        self.w_t = w_t
        self.l_cyclic = l_cyclic # perhaps enforce some guarentees about this being on the gpu?
        
        self.with_dlogp = with_dlogp
        self._cyclic_counter = 0 # TODO: REMOVE THIS?
            
    def forward(self, t, xs): # modified by Ben... TODO: FINISH/FIX THIS...
        """
        Forward pass with cyclic loss incorporated.

        Args:
            t (float): Time, between 0 and 1.
            xs (torch.Tensor): Atomic positions of shape (batch_size, n_particles * n_dimensions).

        Returns:
            torch.Tensor: Updated velocities with cyclic loss included, of the same shape as `xs`.
        """
        #print(t)
        #print(isinstance(t, float))

        n_batch = xs.shape[0]
        
        vel = super().forward(t, xs)

        cyclic_loss = self.l_cyclic(xs.view(n_batch, self._n_particles, self._n_dimension))

        grad_cyclic = torch.autograd.grad(
            cyclic_loss, xs, grad_outputs=torch.ones_like(cyclic_loss), create_graph=True
            )[0].view(n_batch, self._n_particles* self._n_dimension)

        # Scale cyclic loss gradient by w_t(t)
        loss_coeff = self.w_t(t)

        # Add the scaled gradient to the velocity
        vel = (1 - loss_coeff) * vel + loss_coeff * grad_cyclic

        # Reshape velocity back to the original format
        vel = vel.view(n_batch, self._n_particles * self._n_dimension)
        vel = remove_mean(vel) # Center the CoM

        self._cyclic_counter += 1
        return vel.view(n_batch,  self._n_particles* self._n_dimension)

class EGNN_dynamics_AD2_cat_pruned_conditioned(EGNN_dynamics_AD2_cat_bb_all_sc_adjacent): # likely need to clean this code up...
    # Conditioned on time
    # gyration + cyclic conditioning

    '''
    This model should not be used in training.

    A class representing the NetDynamics of all backbone atoms fully connected to one another, but with sidechain atoms only 
    connected fully to other atoms within their, or adjacent, amino acids.

    Parameters:
    ----------
    ...
    '''

    def __init__(self, *args, w_t: LossCoeff, l_cyclic: CyclicLossHandler, g_t: LossCoeff, 
                 l_gyration: GyrationLossHandler, condition_time: bool = True, with_dlogp: bool=True, **kwargs):
        super().__init__(*args, condition_time = condition_time, **kwargs)
        
        self.w_t = w_t
        self.l_cyclic = l_cyclic # perhaps enforce some guarentees about this being on the gpu?

        self.g_t = g_t
        self.l_gyration = l_gyration

        self.l_total = GyrationCyclicLossHandler(l_cyclic= l_cyclic, l_gyration=l_gyration, gamma=g_t)
        
        self.with_dlogp = with_dlogp
       #self._cyclic_counter = 0 # TODO: REMOVE THIS?
            
    def forward(self, t, xs): # modified by Ben... TODO: FINISH/FIX THIS...
        """
        Forward pass with cyclic loss incorporated.

        Args:
            t (float): Time, between 0 and 1.
            xs (torch.Tensor): Atomic positions of shape (batch_size, n_particles * n_dimensions).

        Returns:
            torch.Tensor: Updated velocities with cyclic loss included, of the same shape as `xs`.
        """
        #print(t)
        #print(isinstance(t, float))

        n_batch = xs.shape[0]
        
        vel = super().forward(t, xs)

        total_loss = self.l_total(positions = xs.view(n_batch, self._n_particles, self._n_dimension), t = t)

        grad_conditional = torch.autograd.grad(
            total_loss, xs, grad_outputs=torch.ones_like(total_loss), create_graph=True
            )[0].view(n_batch, self._n_particles* self._n_dimension)

        # Scale cyclic loss gradient by w_t(t)
        loss_coeff = self.w_t(t)

        # Add the scaled gradient to the velocity
        vel = (1 - loss_coeff) * vel + loss_coeff * grad_conditional

        # Reshape velocity back to the original format
        vel = vel.view(n_batch, self._n_particles * self._n_dimension)
        vel = remove_mean(vel) # Center the CoM

       # self._cyclic_counter += 1
        return vel.view(n_batch,  self._n_particles* self._n_dimension) # does this make sense?