from abc import ABC, abstractmethod
import math
import torch

class LossCoeff(ABC):
    '''
    An abstract class to handle loss function coefficients for cyclic conditioning.
    '''
    
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        '''
        Evaluates the loss function coefficient at time t, which must be a 
        torch.tensor of shape [batch_size, ], whose values are ∈ [0, 1].
        
        Will output a torch.Tensor whose vals are also ∈ [0, 1].
        '''
        pass

    @staticmethod
    def _assert_in_range(w_t: torch.Tensor):
        assert torch.all(0 <= w_t <= 1), 'Loss coeff must be between 0, 1.'

class PseudoGaussian(LossCoeff):
    '''
    A class for intantiating loss function coefficients that are of a gaussian shape,
    but whose area is not normalized. Hence, pseudo-gaussian.
    '''

    def __init__(self, mu: float, s: float, coeff: float = 1.0):
        '''
        Initializes an instance of a pseudo-gaussian loss coefficient.

        An instance of a cyclic loss coefficient who represents a non-normalized gaussian
        with a customizable mean, width and height.

        Args:
        ----
        mu (float): the temporal location of the maximum. Must be between 0 & 1
        s (float): the width of the maximum. Must be > 0
        coeff (float): the 
        '''
        
        assert 0 <= mu <= 1, 'mu must be a float between 0 & 1, inclusive'
        assert s > 0, 's must be a positive float'
        assert 0 <= coeff <= 1, 'coeff must be a float between 0 & 1, inclusive'

        self._mu = mu
        self._s = s
        self._coeff = coeff
    
    # getters vvv
    @property
    def mu(self) -> float:
        return self._mu
    
    @property
    def s(self) -> float:
        return self._s
    
    @property
    def coeff(self) -> float:
        return self._coeff
    # getters ^^^

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        f'''
        Evaluates the loss coefficient for a pseudo-gaussian loss of mu: {self.mu},
        s: {self.s}, & coeff: {self.coeff}.

        Args:
        ----
        t: torch.Tensor [batch_size, ]
            the torch.tensor representing that moment in time of the inference.


        Returns:
        -------
        w_t: torch.Tensor [batch_size, ]
            the loss coefficient for that particular moment in time.
        '''

        w_t = self._coeff * torch.exp(-0.5 * ((t - self._mu) / self._s)**2)
        self._assert_in_range(w_t)

        return w_t

class MaxwellBoltzmann(LossCoeff):
    def __init__(self, alpha: float):
        self._alpha = alpha
    
    @property
    def alpha(self) -> float:
        return self._alpha
    
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        w_t = torch.sqrt(2 / math.pi) * (t**2 / self._alpha**3) * torch.exp(-t**2 / (2 * self._alpha**2))
        self._assert_in_range(w_t)

        return w_t
    
class Constant(LossCoeff):
    def __init__(self, const: float):
        assert 0 <= const <= 1, 'const must be a float ∈ [0, 1]'
        self._const = const
    
    @property
    def const(self) -> float:
        return self._const
    
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        w_t = self._const * torch.ones_like(t)
        self._assert_in_range(w_t)

        return w_t

class Pulse(LossCoeff): # TODO: finish implementing this.
    def __init__(self, t1: float, t2: float, const: float):
        assert 0.0 <= t1 < 1.0, 't1 must be ∈ [0, 1)'
        assert 0.0 < t2 <= 1.0, 't2 must be ∈ (0, 1]'
        assert t1 <= t2, 't1 must be <= t2'
        assert 0.0 <= const <= 1.0, 'const must be ∈ [0, 1]'

        self._t1 = t1
        self._t2 = t2
        self._const = const
    
    @property
    def t1(self):
        return self._t1
    
    @property
    def t2(self):
        return self._t2
    
    @property
    def const(self):
        return self._const

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        const_tensor = self.const * torch.ones_like(t, device= t.device)
        zeros_tensor = torch.zeros_like(t, device = t.device)

        w_t = torch.where(self.t1 <= t <= self.t2, const_tensor, zeros_tensor)

        self._assert_in_range(w_t)

        return w_t