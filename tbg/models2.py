import torch
import torch.nn as nn

from tbg.gcl import E_GCL_vel, E_GCL, GCL
from tbg.utils import remove_mean, remove_mean_with_mask, generate_bb_all_sc_adjacent_from_pdb

### MANY OF THESE SEEM TO BE THE DYNAMICS FUNCTIONS USED IN THE BB_DYNAMICS ###

class EGNN_dynamics(nn.Module):
    def __init__(self, n_particles, n_dimension, hidden_nf=64, device='cpu',
            act_fn=torch.nn.SiLU(), n_layers=4, recurrent=True, attention=False,
                 condition_time=True, tanh=False, mode='egnn_dynamics', agg='sum'):
        super().__init__()
        self.mode = mode
        if mode == 'egnn_dynamics':
            self.egnn = EGNN(in_node_nf=1, in_edge_nf=1, hidden_nf=hidden_nf, device=device, act_fn=act_fn, n_layers=n_layers, recurrent=recurrent, attention=attention, tanh=tanh, agg=agg)
        elif mode == 'gnn_dynamics':
            self.gnn = GNN(in_node_nf=1 + n_dimension, in_edge_nf=0, hidden_nf=hidden_nf, out_node_nf=n_dimension, device=device, act_fn=act_fn, n_layers=n_layers, recurrent=recurrent, attention=attention)

        self.device = device
        self._n_particles = n_particles
        self._n_dimension = n_dimension
        self.edges = self._create_edges()
        self._edges_dict = {}
        self.condition_time = condition_time
        # Count function calls
        self.counter = 0

    def forward(self, t, xs):

        n_batch = xs.shape[0]
        edges = self._cast_edges2batch(self.edges, n_batch, self._n_particles)
        edges = [edges[0], edges[1]]
        #Changed by Leon
        x = xs.reshape(n_batch*self._n_particles, self._n_dimension).clone()
        h = torch.ones(n_batch, self._n_particles).to(self.device)
    
        if self.condition_time:
            h = h*t
        h = h.reshape(n_batch*self._n_particles, 1)
        if self.mode == 'egnn_dynamics':
            edge_attr = torch.sum((x[edges[0]] - x[edges[1]])**2, dim=1, keepdim=True)
            _, x_final = self.egnn(h, x, edges, edge_attr=edge_attr)
            vel = x_final - x

        elif self.mode == 'gnn_dynamics':
            h = torch.cat([h, x], dim=1)
            vel = self.gnn(h, edges)

        vel = vel.view(n_batch, self._n_particles, self._n_dimension)
        vel = remove_mean(vel)
        #Changed by Leon
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
    
    
class EGNN_dynamics_transferable(nn.Module):
    def __init__(self, n_particles, n_dimension, hidden_nf=64, device='cpu',
            act_fn=torch.nn.SiLU(), n_layers=4, recurrent=True, attention=False,
                 condition_time=True, tanh=False, mode='egnn_dynamics', agg='sum'):
        super().__init__()
        self.mode = mode
        if mode == 'egnn_dynamics':
            self.egnn = EGNN(in_node_nf=1, in_edge_nf=1, hidden_nf=hidden_nf, device=device, act_fn=act_fn, n_layers=n_layers, recurrent=recurrent, attention=attention, tanh=tanh, agg=agg)
        elif mode == 'gnn_dynamics':
            self.gnn = GNN(in_node_nf=1 + n_dimension, in_edge_nf=0, hidden_nf=hidden_nf, out_node_nf=n_dimension, device=device, act_fn=act_fn, n_layers=n_layers, recurrent=recurrent, attention=attention)

        self.device = device
        self._n_particles = n_particles
        self._n_dimension = n_dimension
        self.edges = self._create_edges()
        self._edges_dict = {}
        self.condition_time = condition_time
        # Count function calls
        self.counter = 0

    def forward(self, t, xs, node_mask, edge_mask):

        n_batch = xs.shape[0]
        #edges = self._cast_edges2batch(self.edges, n_batch, self._n_particles)
        edges = self.get_adj_matrix(self._n_particles, n_batch, self.device)
        edges = [edges[0], edges[1]]
        node_mask = node_mask.view(n_batch*self._n_particles, 1)
        edge_mask = edge_mask.view(n_batch*self._n_particles*self._n_particles, 1)
        #Changed by Leon
        x = xs.reshape(n_batch*self._n_particles, self._n_dimension).clone() * node_mask
        h = torch.ones(n_batch, self._n_particles).to(self.device)
    
        if self.condition_time:
            h = h*t
        h = h.reshape(n_batch*self._n_particles, 1)
        if self.mode == 'egnn_dynamics':
            edge_attr = torch.sum((x[edges[0]] - x[edges[1]])**2, dim=1, keepdim=True)
            _, x_final = self.egnn(h, x, edges, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
            vel = (x_final - x) * node_mask  # This masking operation is redundant but just in case

        elif self.mode == 'gnn_dynamics':
            h = torch.cat([h, x], dim=1)
            vel = self.gnn(h, edges)

        vel = vel.view(n_batch, self._n_particles, self._n_dimension)
        vel = remove_mean_with_mask(vel, node_mask.view(n_batch, self._n_particles, 1))
        #Changed by Leon
        self.counter += 1
        return vel.view(n_batch, self._n_particles* self._n_dimension)

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
    
    def get_adj_matrix(self, n_nodes, batch_size, device):
        if n_nodes in self._edges_dict:
            edges_dic_b = self._edges_dict[n_nodes]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
            else:
                # get edges for a single sample
                rows, cols = [], []
                for batch_idx in range(batch_size):
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            rows.append(i + batch_idx * n_nodes)
                            cols.append(j + batch_idx * n_nodes)
        else:
            self._edges_dict[n_nodes] = {}
            return self.get_adj_matrix(n_nodes, batch_size, device)

        edges = [torch.LongTensor(rows).to(device), torch.LongTensor(cols).to(device)]
        return edges

class EGNN_dynamics_transferable_MD(nn.Module):
    def __init__(self, n_particles, n_dimension, h_size, hidden_nf=64, device='cpu',
            act_fn=torch.nn.SiLU(), n_layers=4, recurrent=True, attention=False,
                 condition_time=True, tanh=False, mode='egnn_dynamics', agg='sum'):
        super().__init__()
        self.mode = mode
        if mode == 'egnn_dynamics':
            self.egnn = EGNN(in_node_nf=h_size+1 if condition_time else h_size, in_edge_nf=1, hidden_nf=hidden_nf, device=device, act_fn=act_fn, n_layers=n_layers, recurrent=recurrent, attention=attention, tanh=tanh, agg=agg)
        else:
            raise NotImplemented()
            
        self.device = device
        # maximum number of possible particles
        self._n_particles = n_particles
        self._n_dimension = n_dimension
        self.edges = self._create_edges()
        self._edges_dict = {}
        self.condition_time = condition_time
        # Count function calls
        self.counter = 0

    def forward(self, t, xs, h, node_mask, edge_mask):

        n_batch = xs.shape[0]
        #edges = self._cast_edges2batch(self.edges, n_batch, self._n_particles)
        edges = self.get_adj_matrix(self._n_particles, n_batch, self.device)
        edges = [edges[0], edges[1]]

        node_mask = node_mask.view(n_batch*self._n_particles, 1)
        edge_mask = edge_mask.view(n_batch*self._n_particles*self._n_particles, 1)
        #Changed by Leon
        x = xs.reshape(n_batch*self._n_particles, self._n_dimension).clone() * node_mask
        # apply the node mask here just in case
        h = h.reshape(n_batch*self._n_particles, -1).to(self.device) * node_mask
        t = torch.tensor(t).to(xs)
        if t.shape != (n_batch, 1):
            t = t.repeat(n_batch)
        t = t.repeat(1, self._n_particles)
        t = t.reshape(n_batch*self._n_particles, 1) * node_mask
    
        if self.condition_time:
            h = torch.cat([h, t], dim=-1)
        h = h.reshape(n_batch*self._n_particles, -1)
        if self.mode == 'egnn_dynamics':
            edge_attr = torch.sum((x[edges[0]] - x[edges[1]])**2, dim=1, keepdim=True)
            _, x_final = self.egnn(h, x, edges, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
            vel = (x_final - x) * node_mask  # This masking operation is redundant but just in case

        else:
            raise NotImplemented()


        vel = vel.view(n_batch, self._n_particles, self._n_dimension)
        vel = remove_mean_with_mask(vel, node_mask.view(n_batch, self._n_particles, 1))
        #Changed by Leon
        self.counter += 1
        return vel.view(n_batch, self._n_particles* self._n_dimension)

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
    # Do we need this?????
    def get_adj_matrix(self, n_nodes, batch_size, device):
        if n_nodes in self._edges_dict:
            edges_dic_b = self._edges_dict[n_nodes]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
            else:
                # get edges for a single sample
                rows, cols = [], []
                for batch_idx in range(batch_size):
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            rows.append(i + batch_idx * n_nodes)
                            cols.append(j + batch_idx * n_nodes)
        else:
            self._edges_dict[n_nodes] = {}
            return self.get_adj_matrix(n_nodes, batch_size, device)

        edges = [torch.LongTensor(rows).to(device), torch.LongTensor(cols).to(device)]
        return edges
    
    
class EGNN_dynamics_AD2(nn.Module):
    def __init__(self, n_particles, n_dimension,h_initial, hidden_nf=64, device='cpu',
            act_fn=torch.nn.SiLU(), n_layers=4, recurrent=True, attention=False,
                 condition_time=True, tanh=False, mode='egnn_dynamics', agg='sum'):
        super().__init__()
        self.mode = mode
        # Initial one hot encoding of the different element types
        self.h_initial = h_initial

        if mode == 'egnn_dynamics':
            self.egnn = EGNN(in_node_nf=h_initial.size(1), in_edge_nf=1, hidden_nf=hidden_nf, device=device, act_fn=act_fn, n_layers=n_layers, recurrent=recurrent, attention=attention, tanh=tanh, agg=agg)
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
        

    def forward(self, t, xs, node_mask, edge_mask):

        n_batch = xs.shape[0]
        edges = self._cast_edges2batch(self.edges, n_batch, self._n_particles)
        edges = [edges[0], edges[1]]
        node_mask = node_mask.view(n_batch*self._n_particles, 1)
        edge_mask = edge_mask.view(n_batch*self._n_particles*self._n_particles, 1)
        #Changed by Leon
        x = xs.reshape(n_batch*self._n_particles, self._n_dimension).clone()* node_mask
        h = self.h_initial.to(self.device).reshape(1,-1)
        h = h.repeat(n_batch, 1)
        #print(t.shape, h.shape)

        if self.condition_time:
            h = h*t
        h = h.reshape(n_batch*self._n_particles, -1)
        if self.mode == 'egnn_dynamics':
            edge_attr = torch.sum((x[edges[0]] - x[edges[1]])**2, dim=1, keepdim=True)
            _, x_final = self.egnn(h, x, edges, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
            vel = (x_final - x) * node_mask  # This masking operation is redundant but just in case

        else:
            raise NotImplemented()
            
        vel = vel.view(n_batch, self._n_particles, self._n_dimension)
        vel = remove_mean_with_mask(vel, node_mask.view(n_batch, self._n_particles, 1))
        #Changed by Leon
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

class EGNN_dynamics_AD2_cat(nn.Module):
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
            
            self.egnn = EGNN(in_node_nf=h_size, in_edge_nf=1, hidden_nf=hidden_nf, device=device, act_fn=act_fn, n_layers=n_layers, recurrent=recurrent, attention=attention, tanh=tanh, agg=agg)
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

class EGNN_dynamics_AD2_cat_bb_all_sc_adjacent(EGNN_dynamics_AD2_cat): # created by Ben
    def __init__(self, *args, pdb_file=None, **kwargs):
        self.counter = 0 ### feel free to remove later... for bugtesting.
        self._custom_adj_matrix = None

        super().__init__(*args, **kwargs)
        self.pdb_file = pdb_file
        if pdb_file is not None: #maybe you NEED to specify the pdb file, so maybe take this out...?
            # Generate a custom adjacency matrix from the PDB file
            edge_tensor = generate_bb_all_sc_adjacent_from_pdb(pdb_file)
            self._custom_adj_matrix = edge_tensor
            self.edges = edge_tensor
        else:
            self.edges = super()._create_edges()

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
    
#    def forward(self, t, xs):
#        return super().forward(t, xs)

class EGNN_dynamics_AD2_cat_bb_all_sc_adj_cyclic(EGNN_dynamics_AD2_cat_bb_all_sc_adjacent): # likely need to clean this code up...
    '''
    This model should not be used in training.

    A class representing the NetDynamics of all backbone atoms fully connected to one another, but with sidechain atoms only 
    connected fully to other atoms within their, or adjacent, amino acids.

    Parameters:
    ----------
    l_cyclic (function: torch.Tensor -> torch.Tensor): 
        a function which takes a tensor of shape (n_batch, n_atoms, n_dimensions), representing the atom positions, 
        at that point in time, accross batched, and which outputs a single number [torch.Tensor, shape (n_batch), )], 
        representing the loss associated with cyclicality. Must be a positive number for all batches. All computation 
        for this should be done on the gpu.

    w_t (function: torch.Tensor -> torch.Tensor):
        a function of a torch.Tensor between 0 & 1 [torch.Tensor, shape (1, )] representing time, which outputs another 
        float between 0 & 1, representing the loss coefficient [torch.Tensor, shape (n_batch, )]. Ideally this is batch 
        friendly and does all computation on the gpu.
    '''

    def __init__(self, *args, w_t = None, l_cyclic=None, with_dlogp=True, **kwargs):
        super().__init__(*args, **kwargs)
        
        if w_t is None: ### move this function onto the gpu, if you can.
            self.w_t = lambda t: torch.tensor(0.0 * t, device=self.device)
        else:
            self.w_t = w_t # perhaps enforce some guarentees about this being on the gpu?
        
        if l_cyclic is None: ### move this function onto the gpu, if you can.
            # TODO: review. will this loss do nothing?
            self.l_cyclic = lambda x: (0.01 * x ** 2).sum(dim=(1, 2)) # make this something which does nothing
        else:
            self.l_cyclic = l_cyclic # perhaps enforce some guarentees about this being on the gpu?
        
        self.with_dlogp = with_dlogp
        self._cyclic_counter = 0 # TODO: REMOVE THIS?
            
    def forward(self, t, xs): # modified by Ben... TODO: FINISH/FIX THIS...
        """
        Forward pass with cyclic loss incorporated.

        Args:
            t (float or torch.Tensor): Time, between 0 and 1.
            xs (torch.Tensor): Atomic positions of shape (batch_size, n_particles * n_dimensions).

        Returns:
            torch.Tensor: Updated velocities with cyclic loss included, of the same shape as `xs`.
        """

        n_batch = xs.shape[0]
        vel = super().forward(t, xs) # disable grad for this?

        if torch.any(torch.isnan(xs)): #huh... not working???
            print('--------========== ISNAN CONDITION TRIGGERED ==========--------')
            return vel

        # Now re-enable gradient tracking for cyclic loss
        xs.requires_grad_(True) # will this work? we shall see.
        cyclic_loss = self.l_cyclic(xs.view(n_batch, self._n_particles, self._n_dimension)) ### TODO: ahh, this seems to create the error...

        grad_cyclic = torch.autograd.grad( ### see if this works...
            cyclic_loss, xs, grad_outputs=torch.ones_like(cyclic_loss), create_graph=True
            )[0].view(n_batch,  self._n_particles* self._n_dimension)

        # Scale cyclic loss gradient by w_t(t)
        loss_coefficient = self.w_t(t)

        print

        # Add the scaled gradient to the velocity
        vel = (1 - loss_coefficient) * vel + loss_coefficient * grad_cyclic

        # Reshape velocity back to the original format
        vel = vel.view(n_batch, self._n_particles * self._n_dimension)
        vel = remove_mean(vel) # Center the CoM

        self._cyclic_counter += 1
        return vel.view(n_batch,  self._n_particles* self._n_dimension)

class EGNN_dynamics_QM9(nn.Module):
    def __init__(self, in_node_nf, context_node_nf,
                 n_dims, hidden_nf=64, device='cpu',
            act_fn=torch.nn.SiLU(), n_layers=4, recurrent=True, attention=False,
                 condition_time=True, tanh=False, mode='egnn_dynamics', agg='sum'):
        super().__init__()
        self.mode = mode
        if mode == 'egnn_dynamics':
            self.egnn = EGNN(
                in_node_nf=in_node_nf + context_node_nf, in_edge_nf=1,
                hidden_nf=hidden_nf, device=device, act_fn=act_fn,
                n_layers=n_layers, recurrent=recurrent, attention=attention, tanh=tanh, agg=agg)
            self.in_node_nf = in_node_nf
        elif mode == 'gnn_dynamics':
            self.gnn = GNN(in_node_nf=in_node_nf + context_node_nf + 3, in_edge_nf=0, hidden_nf=hidden_nf, out_node_nf= 3 + in_node_nf, device=device, act_fn=act_fn, n_layers=n_layers, recurrent=recurrent, attention=attention)

        self.context_node_nf = context_node_nf
        self.device = device
        self.n_dims = n_dims
        self._edges_dict = {}
        self.condition_time = condition_time

    def forward(self, t, xh, node_mask, edge_mask, context=None):
        raise NotImplementedError

    def wrap_forward(self, node_mask, edge_mask, context):
        def fwd(time, state):
            return self._forward(time, state, node_mask, edge_mask, context)
        return fwd

    def unwrap_forward(self):
        return self._forward

    def _forward(self, t, xh, node_mask, edge_mask, context):
        bs, n_nodes, dims = xh.shape
        h_dims = dims - self.n_dims
        edges = self.get_adj_matrix(n_nodes, bs, self.device)
        node_mask = node_mask.view(bs*n_nodes, 1)
        edge_mask = edge_mask.view(bs*n_nodes*n_nodes, 1)
        xh = xh.view(bs*n_nodes, -1).clone() * node_mask
        x = xh[:, 0:self.n_dims].clone()
        if h_dims == 0:
            h = torch.ones(bs*n_nodes, 1).to(self.device)
        else:
            h = xh[:, self.n_dims:].clone()

        if self.condition_time:
            h_time = torch.empty_like(h[:, 0:1]).fill_(t)
            h = torch.cat([h, h_time], dim=1)

        if context is not None:
            # We're conditioning, awesome!
            context = context.view(bs*n_nodes, self.context_node_nf)
            h = torch.cat([h, context], dim=1)
            
        if self.mode == 'egnn_dynamics':
            edge_attr = torch.sum((x[edges[0]] - x[edges[1]]) ** 2, dim=1, keepdim=True)
            h_final, x_final = self.egnn(h, x, edges, node_mask=node_mask, edge_mask=edge_mask, edge_attr=edge_attr)
            vel = (x_final - x) * node_mask  # This masking operation is redundant but just in case
        elif self.mode == 'gnn_dynamics':
            xh = torch.cat([x, h], dim=1)
            output = self.gnn(xh, edges, node_mask=node_mask)
            vel = output[:, 0:3] * node_mask
            h_final = output[:, 3:]

        else:
            raise Exception("Wrong mode %s" % self.mode)

        if context is not None:
            # Slice off context size:
            h_final = h_final[:, :-self.context_node_nf]

        if self.condition_time:
            # Slice off last dimension which represented time.
            h_final = h_final[:, :-1]

        vel = vel.view(bs, n_nodes, -1)

        if node_mask is None:
            vel = remove_mean(vel)
        else:
            vel = remove_mean_with_mask(vel, node_mask.view(bs, n_nodes, 1))

        if h_dims == 0:
            return vel
        else:
            h_final = h_final.view(bs, n_nodes, -1)
            return torch.cat([vel, h_final], dim=2)

    def get_adj_matrix(self, n_nodes, batch_size, device):
        if n_nodes in self._edges_dict:
            edges_dic_b = self._edges_dict[n_nodes]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
            else:
                # get edges for a single sample
                rows, cols = [], []
                for batch_idx in range(batch_size):
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            rows.append(i + batch_idx * n_nodes)
                            cols.append(j + batch_idx * n_nodes)
        else:
            self._edges_dict[n_nodes] = {}
            return self.get_adj_matrix(n_nodes, batch_size, device)

        edges = [torch.LongTensor(rows).to(device), torch.LongTensor(cols).to(device)]
        return edges


class EGNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4, recurrent=True, attention=False, norm_diff=True, out_node_nf=None, tanh=False, coords_range=15, agg='sum'):
        super(EGNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range)/self.n_layers
        if agg == 'mean':
            self.coords_range_layer = self.coords_range_layer * 19
        #self.reg = reg
        ### Encoder
        #self.add_module("gcl_0", E_GCL(in_node_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, act_fn=act_fn, recurrent=False, coords_weight=coords_weight))
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, act_fn=act_fn, recurrent=recurrent, attention=attention, norm_diff=norm_diff, tanh=tanh, coords_range=self.coords_range_layer, agg=agg))

        self.to(self.device)

    def forward(self, h, x, edges, edge_attr=None, node_mask=None, edge_mask=None):
        # Edit Emiel: Remove velocity as input
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
        h = self.embedding_out(h)

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h, x


class GNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4, recurrent=True, attention=False, out_node_nf=None):
        super(GNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        ### Encoder
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, act_fn=act_fn, recurrent=recurrent, attention=attention))

        self.to(self.device)

    def forward(self, h, edges, edge_attr=None, node_mask=None, edge_mask=None):
        # Edit Emiel: Remove velocity as input
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edges, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
        h = self.embedding_out(h)

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h
