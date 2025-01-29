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
    def __call__(self, t: float) -> float:
        '''
        Evaluates the loss function coefficient at time t, which must be a float ∈
        [0, 1]. Will output a float also ∈ [0, 1].
        '''
        pass

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

    def __call__(self, t: float) -> float:
        f'''
        Evaluates the loss coefficient for a pseudo-gaussian loss of mu: {self.mu},
        s: {self.s}, & coeff: {self.coeff}.

        Args:
        ----
        t (float): the float representing that moment in time of the inference.

        Returns:
        -------
        w_t (float): the loss coefficient for that particular moment in time.
        '''

        w_t = self._coeff * torch.exp(-0.5 * ((t - self._mu) / self._s)**2)
        assert 0 <= w_t <= 1, '__call__ cannot return a float not ∈ [0, 1], but is trying to'

        return w_t

class MaxwellBoltzmann(LossCoeff):
    def __init__(self, alpha: float):
        self._alpha = alpha
    
    @property
    def alpha(self) -> float:
        return self._alpha
    
    def __call__(self, t: float) -> float:
        w_t = torch.sqrt(2 / math.pi) * (t**2 / self._alpha**3) * torch.exp(-t**2 / (2 * self._alpha**2))
        assert 0 < w_t < 1, '__call__ cannot return a float not ∈ [0, 1], but is trying to'

        return w_t
    
class Constant(LossCoeff):
    def __init__(self, const: float):
        assert 0 <= const <= 1, 'const must be a float ∈ [0, 1]'
        self._const = const
    
    @property
    def const(self) -> float:
        return self._const
    
    def __call__(self, t: float) -> float:
        return self._const

class HeavisideStep(LossCoeff):
    def __init__(self, const):
        pass


class Pulse(LossCoeff): # TODO: finish implementing this.
    def __init__(self, t1: float, t2: float, const: float):
        assert 0 <= t1 < 1, 't1 must be ∈ [0, 1)'
        assert 0 < t2 <= 1, 't2 must be ∈ (0, 1]'
        assert 0 <= const <= 1, 'const must be ∈ [0, 1]'

        self._t1 = t1
        self._t2 = t2
        self._const = const

    def __call__(self, t: float) -> float:
        pass