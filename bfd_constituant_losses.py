
import torch

### MAKING ALL OF THESE BATCH FRIENDLY (ABLE TO BE DONE ACROSS-BATCHES) ###

def sq_distance(a1, a2):
    """
    Compute squared distances for batches of points.

    Parameters:
    a1, a2 (torch.Tensor): Tensors of shape (batch_size, 3) or (n_atoms, 3).

    Returns:
    torch.Tensor: Squared distances of shape (batch_size).
    """
    return torch.sum((a1 - a2) ** 2, dim=-1)

def bond_angle(a1, a2, a3):
    """
    Compute bond angles for batches of points.

    Parameters:
    a1, a2, a3 (torch.Tensor): Tensors of shape (batch_size, 3).

    Returns:
    torch.Tensor: Bond angles of shape (batch_size).
    """
    v1 = a1 - a2
    v2 = a3 - a2

    # Normalize vectors
    v1_norm = v1 / torch.norm(v1, dim=-1, keepdim=True)
    v2_norm = v2 / torch.norm(v2, dim=-1, keepdim=True)

    # Compute cosine of angles
    cos_theta = torch.sum(v1_norm * v2_norm, dim=-1).clamp(-1.0, 1.0)

    return torch.acos(cos_theta)

def dihedral_angle(a1, a2, a3, a4):
    """
    Compute dihedral angles for batches of points.

    Parameters:
    a1, a2, a3, a4 (torch.Tensor): Tensors of shape (batch_size, 3).

    Returns:
    torch.Tensor: Dihedral angles of shape (batch_size).
    """
    v1 = a2 - a1
    v2 = a3 - a2
    v3 = a4 - a3

    # Normal vectors to planes
    n1 = torch.cross(v1, v2, dim=-1)
    n2 = torch.cross(v2, v3, dim=-1)

    # Normalize normal vectors
    n1_norm = n1 / torch.norm(n1, dim=-1, keepdim=True)
    n2_norm = n2 / torch.norm(n2, dim=-1, keepdim=True)

    # Dot product for cosine of dihedral angle
    cos_theta = torch.sum(n1_norm * n2_norm, dim=-1).clamp(-1.0, 1.0)

    # Compute the sign using a helper vector
    m1 = torch.cross(n1_norm, n2_norm, dim=-1)
    sign = torch.sign(torch.sum(m1 * v2, dim=-1))

    return sign * torch.acos(cos_theta)

### ACTUAL LOSS FUNCTIONS DOWN HERE: ###
### NEED TO IMPLEMENT SOME STEEPNESS PARAMETER TO PLAY AROUND WITH ###

def dihedral_angle(a1, a2, a3, a4): #TODO: add tolerance
    """
    Compute dihedral angles for batches of points.

    Parameters:
    a1, a2, a3, a4 (torch.Tensor): Tensors of shape (batch_size, 3).

    Returns:
    torch.Tensor: Dihedral angles of shape (batch_size).
    """
    v1 = a2 - a1
    v2 = a3 - a2
    v3 = a4 - a3

    # Normal vectors to planes
    n1 = torch.cross(v1, v2, dim=-1)
    n2 = torch.cross(v2, v3, dim=-1)

    # Normalize normal vectors
    n1_norm = n1 / torch.norm(n1, dim=-1, keepdim=True)
    n2_norm = n2 / torch.norm(n2, dim=-1, keepdim=True)

    # Dot product for cosine of dihedral angle
    cos_theta = torch.sum(n1_norm * n2_norm, dim=-1).clamp(-1.0, 1.0)

    # Compute the sign using a helper vector
    m1 = torch.cross(n1_norm, n2_norm, dim=-1)
    sign = torch.sign(torch.sum(m1 * v2, dim=-1))

    return sign * torch.acos(cos_theta)

def bond_angle_loss(a1, a2, a3, target_angle): #TODO: add tolerance?
    """
    Compute bond angle losses for batches.

    Parameters:
    a1, a2, a3 (torch.Tensor): Tensors of shape (batch_size, 3).
    target_angle (float): Target bond angle in radians.

    Returns:
    torch.Tensor: Losses of shape (batch_size).
    """
    b_angle = bond_angle(a1, a2, a3)
    return (b_angle - target_angle) ** 2


def dihedral_angle_loss_(a1, a2, a3, a4, target_angle):
    """
    Compute dihedral angle losses for batches.

    Parameters:
    a1, a2, a3, a4 (torch.Tensor): Tensors of shape (batch_size, 3).
    target_angle (float): Target dihedral angle in radians.

    Returns:
    torch.Tensor: Losses of shape (batch_size).
    """
    d_angle = dihedral_angle(a1, a2, a3, a4)
    return (d_angle - target_angle) ** 2

def motif_absolute(*argv, target_structure): ### NEED TO IMPLEMENT
    """
    Compute a rotationally invariant loss based on relative structures.

    Parameters:
    positions (torch.Tensor): Tensor of shape (N, 3) representing current particle positions.
    reference_positions (torch.Tensor): Tensor of shape (N, 3) representing the reference structure.

    Returns:
    torch.Tensor: Scalar loss value.
    """
    
    pass

def soft_min(inputs, alpha=-10):
    """
    Compute soft minimum across batches; as alpha -> -inf, becomes a hard minimum. As alpha -> 0, becomes
    a simple average. as alpha -> +inf, becomes a hard maximum

    Parameters:
    inputs (torch.Tensor): Tensor of shape (batch_size, n_losses).
    alpha (float): Smoothness parameter.

    Returns:
    torch.Tensor: Soft minimum for each batch of shape (batch_size).
    """
    exps = torch.exp(alpha * inputs)
    return torch.sum(inputs * exps, dim=-1) / torch.sum(exps, dim=-1)
