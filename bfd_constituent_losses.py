
import torch

# batch-friendly loss building blocks

def sq_distance(a1, a2):
    """
    Compute squared distances for batches of points.

    Parameters:
    a1, a2 (torch.Tensor): Tensors of shape (batch_size, 3) or (n_atoms, 3).

    Returns:
    torch.Tensor: Squared distances of shape (batch_size).
    """

    result = torch.sum((a1 - a2) ** 2, dim=-1)
    assert torch.all(torch.isfinite(result)), f"sq_distance yielded something non_finite! result: {result}"
    
    return result

def bond_angle(a1, a2, a3):
    """
    Compute bond angles for batches of points.

    Parameters:
    a1, a2, a3 (torch.Tensor): Tensors of shape (batch_size, 3).

    Returns:
    torch.Tensor: Bond angles of shape (batch_size), in radians.
    """
    v1 = a1 - a2
    v2 = a3 - a2

    # Normalize vectors
    v1_norm = v1 / torch.norm(v1, dim=-1, keepdim=True)### BUG TESTING
    v2_norm = v2 / torch.norm(v2, dim=-1, keepdim=True) ### BUG TESTING

    # Compute cosine of angles
    cos_theta = torch.sum(v1_norm * v2_norm, dim=-1).clamp(-1.0, 1.0)
    result = torch.acos(cos_theta)

    assert torch.all(torch.isfinite(result)), f"bond_angle yielded something non_finite! result: {result}"

    return result

def dihedral_angle(a1, a2, a3, a4):
    """
    Compute dihedral angles for batches of points.

    Parameters:
    a1, a2, a3, a4 (torch.Tensor): Tensors of shape (batch_size, 3).

    Returns:
    torch.Tensor: Dihedral angles of shape (batch_size), in radians.
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

    result = sign * torch.acos(cos_theta)

    assert torch.all(torch.isfinite(result)), f"dihedral_angle yielded something non finite! result: {result}"

    return result

def distance_loss(a1, a2, target_distance, tolerance=0.0):
    """
    Compute distance losses with optional tolerance for batches.

    Parameters:
    a1, a2 (torch.Tensor): Tensors of shape (batch_size, 3).
    target_distance (float): Target bond distance.
    tolerance (float): No-penalty range around the target distance.

    Returns:
    torch.Tensor: Losses of shape (batch_size).
    """
    sq_dist = sq_distance(a1, a2)
    assert torch.all(sq_dist >= 0), f"Negative squared distance found! a1: {a1}, a2: {a2}, sq_dist: {sq_dist}"
    dist = torch.sqrt(sq_dist)
    error = torch.abs(dist - target_distance)
    
    # Apply tolerance: zero penalty within tolerance range
    penalty = torch.where(error <= tolerance, torch.zeros_like(error), (error - tolerance) ** 2)

    result = penalty
    assert torch.all(torch.isfinite(result)), f"distance_loss yielded something non finite! result: {result}"

    return result

def bond_angle_loss(a1, a2, a3, target_angle, tolerance=0.0):
    """
    Compute bond angle losses with tolerance for batches.

    Parameters:
    a1, a2, a3 (torch.Tensor): Tensors of shape (batch_size, 3).
    target_angle (float): Target bond angle in radians.
    tolerance (float): No-penalty range around the target angle.

    Returns:
    torch.Tensor: Losses of shape (batch_size).
    """
    b_angle = bond_angle(a1, a2, a3)
    error = torch.abs(b_angle - target_angle)
    
    # Apply tolerance
    penalty = torch.where(error <= tolerance, torch.zeros_like(error), (error - tolerance) ** 2)

    result = penalty
    assert torch.all(torch.isfinite(result)), f"bond_angle_loss yielded something non finite! result: {result}"

    return result

def dihedral_angle_loss(a1, a2, a3, a4, target_angle, tolerance=0.0):
    """
    Compute dihedral angle losses with tolerance for batches.

    Parameters:
    a1, a2, a3, a4 (torch.Tensor): Tensors of shape (batch_size, 3).
    target_angle (float): Target dihedral angle in radians.
    tolerance (float): No-penalty range around the target angle.

    Returns:
    torch.Tensor: Losses of shape (batch_size).
    """
    d_angle = dihedral_angle(a1, a2, a3, a4)
    error = torch.abs(d_angle - target_angle)
    
    # Apply tolerance
    penalty = torch.where(error <= tolerance, torch.zeros_like(error), (error - tolerance) ** 2)

    result = penalty
    assert torch.all(torch.isfinite(result)), f"dihedral_angle_loss yielded something non finite! result: {result}"

    return result

def motif_absolute(*argv, target_structure):
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
    inputs (torch.Tensor): Tensor of shape (n_batch, n_losses).
    alpha (float): Smoothness parameter.

    Returns:
    torch.Tensor: Soft minimum for each batch of shape (batch_size).
    """

    assert torch.all(torch.isfinite(inputs)), "Inputs to soft_min contain non-finite values!"
    assert torch.all(inputs > 0), "All inputs to soft_min must be positive."

    exps = torch.exp(alpha * inputs)
    return torch.sum(inputs * exps, dim=-1) / torch.sum(exps, dim=-1)
