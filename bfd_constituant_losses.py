
import torch
import mdtraj as md

def sq_distance(a1, a2):
    a1 = torch.tensor(a1, requires_grad=True, device="cuda" if torch.cuda.is_available() else "cpu")
    a2 = torch.tensor(a2, requires_grad=True, device="cuda" if torch.cuda.is_available() else "cpu")

    # Calculate squared distance
    return torch.sum((a1 - a2) ** 2)

def bond_angle(a1, a2, a3):
    # Convert inputs to PyTorch tensors if they are not already
    a1 = torch.tensor(a1, requires_grad=True, device="cuda" if torch.cuda.is_available() else "cpu")
    a2 = torch.tensor(a2, requires_grad=True, device="cuda" if torch.cuda.is_available() else "cpu")
    a3 = torch.tensor(a3, requires_grad=True, device="cuda" if torch.cuda.is_available() else "cpu")
    
    v1 = a1 - a2
    v2 = a3 - a2

    # Normalize vectors to get unit vectors
    v1_norm = v1 / torch.norm(v1)
    v2_norm = v2 / torch.norm(v2)

    # Compute the cosine of the angle using dot product
    cos_theta = torch.dot(v1_norm, v2_norm).clamp(-1.0, 1.0)  # Clamp to avoid numerical errors out of range

    # Compute the angle in radians
    angle = torch.acos(cos_theta)

    return angle

def dihedral_angle(a1, a2, a3, a4): ### GO OVER THE INNARDS OF THIS FUNCTION! ###
    # Convert inputs to PyTorch tensors if they are not already
    if not isinstance(a1, torch.Tensor):
        a1 = torch.tensor(a1, requires_grad=True, device="cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(a2, torch.Tensor):
        a2 = torch.tensor(a2, requires_grad=True, device="cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(a3, torch.Tensor):
        a3 = torch.tensor(a3, requires_grad=True, device="cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(a4, torch.Tensor):
        a4 = torch.tensor(a4, requires_grad=True, device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Calculate vectors
    v1 = a2 - a1
    v2 = a3 - a2
    v3 = a4 - a3

    # Calculate normal vectors of the planes formed by (a1, a2, a3) and (a2, a3, a4)
    n1 = torch.cross(v1, v2)
    n2 = torch.cross(v2, v3)

    # Normalize the normal vectors
    n1_norm = n1 / torch.norm(n1)
    n2_norm = n2 / torch.norm(n2)

    # Calculate the angle between the planes (dihedral angle) using the dot product
    cos_theta = torch.dot(n1_norm, n2_norm).clamp(-1.0, 1.0)  # Clamp to avoid numerical issues

    # Calculate the sign of the dihedral angle using the direction of v2
    m1 = torch.cross(n1, n2)
    sign = torch.sign(torch.dot(m1, v2))

    # Compute the angle in radians and apply the sign
    angle = sign * torch.acos(cos_theta)

    return angle

### ACTUAL LOSS FUNCTIONS DOWN HERE: ###
### NEED TO IMPLEMENT SOME STEEPNESS PARAMETER TO PLAY AROUND WITH ###

def cyclic_distance_loss(a1, a2, target_distance: float): ### I'd like to add some parameter here to either
    # 1.) not really penalize distances with some range of what is plausible or 2.) to encode a parabolic
    # steepness parameter which encodes the distances over which the "corrective" derivatives arent that 
    # steep (ie: the range of errors that we don't really care about)

    dist = torch.sqrt(sq_distance(a1, a2))

    return (dist - target_distance) ** 2

def bond_angle_loss(a1, a2, a3, target_angle: float):
    b_angle = bond_angle(a1, a2, a3)

    return (b_angle - target_angle) ** 2

def dihedral_angle_loss(a1, a2, a3, a4, target_angle: float):
    d_angle = dihedral_angle(a1, a2, a3, a4)

    return (d_angle - target_angle) ** 2

def soft_min(*argv, alpha = -10): # this will take losses as its input (in the argv part)
    # Stack the input tensors for efficient operations

    # all arvs must be greater than or equal to zero.

    # Seems to be working properly. Only gradient descent will tell...

    inputs = torch.stack(argv)

    # Ensure all inputs are on the same device (move to CUDA if needed)
    if inputs.device != torch.device('cuda'):
        inputs = inputs.to('cuda')

    exps = torch.exp(alpha * inputs)

    soft_min_result = torch.sum(inputs * exps) / torch.sum(exps)

    return soft_min_result

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