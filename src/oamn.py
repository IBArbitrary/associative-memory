"""
Oscillatory Associative Memory Network (OAMN) module.
Based on Nishikawa et al. (2004) (doi: 10.1016/j.physd.2004.06.011)
"""

# imports
# import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import eigsh
from tqdm import trange

class OAMN:
    """
    Class for simulating oscillatory associative memory network

    Attributes
    ----------
    n : int
        Number of oscillators
    p : int
        Number of patterns in memory
    eps : float
        Strength of the second fourier mode
    XiM : np.ndarray
        (p, n) array of memory patterns
    XiMN : list
        List of memory patterns in integer representation
    C : np.ndarray
        (n, n) array of oscillator coupling strengths
    Ct : np.ndarray
        (p, p) array of memory pattern correlations
    """

    def __init__(self, n: int, p: int, eps: float = 0, init: bool = True):
        """
        Parameters
        ----------
        n : int
            Number of oscillators
        p : int
            Number of patterns in memory
        eps : float
            Strength of the second fourier mode
        init : bool
            Initialise memory patterns if True  (default: True)
        """
        self.n = n
        self.p = p
        self.eps = eps
        self.XiM = None
        self.XiMN = None
        self.C = None
        self.Ct = None
        if init:
            self.init_patterns()

    def uni_p(self):
        """
        Generates an uniform random number out of {-1, 1}

        Returns
        -------
        int number
        """
        return 1 if np.random.uniform() > 0.5 else -1

    def pattern(self):
        """
        Generates a pattern of length n

        Returns
        -------
        numpy array of length n for the binary pattern
        """
        n = self.n
        xi_ = []
        for _ in range(n):
            xi_.append(self.uni_p())
        return np.array(xi_)

    def int_rep(self, pattern: np.ndarray):
        """
        Calculates the integer representation of a binary pattern.

        Parameters
        ----------
        pattern : np.ndarray
            The binary pattern

        Returns
        -------
        int value of the integer representation
        """
        return int("".join([str(_) for _ in (pattern+1)//2]), 2)

    def nonmem_pattern(self):
        """
        Generates a random pattern (exclusive of the memory patterns)

        Returns
        -------
        numpy array of length n for the binary pattern
        """
        p1 = self.pattern()
        p1N = self.int_rep(p1)
        if self.XiM is None:
            self.Xi()
        while True:
            if p1N not in self.XiMN:
                break
            else:
                p1 = self.pattern()
        return p1

    def kbiterror_pattern(self, eta: np.ndarray, k: int = 1):
        """
        Randomly flips k bits in the input pattern eta

        Parameters
        ----------
        k : int
            Number of error bits (defualt: 1)
        eta : np.ndarray
            Input pattern

        Returns
        -------
        numpy array of length n for the binary pattern
        """
        n = len(eta)
        eta_ = np.copy(eta)
        inds = np.random.choice(range(n), k, replace=False)
        for _ in inds:
            eta_[_] *= -1
        return eta_

    def smixture_pattern(self, s: int = 3):
        """
        Generates a s-mixture pattern from the memory patterns

        Parameters
        ----------
        s : int
            Number of memory patterns to mix (default: 3)

        Returns
        -------
        numpy array of length n for the binary pattern
        """
        if self.XiM is None:
            self.init_patterns()
        XiM = self.XiM
        eta_s = XiM[:s]
        n = self.n
        eta = np.zeros(n)
        for _ in range(s):
            eta += eta_s[_]
        eta = np.array([_//np.abs(_) for _ in eta])
        return eta

    def Xi(self):
        """
        Generates the n x p matrix Xi of the p-patterns to be memorised,
        of length n each

        Returns
        -------
        numpy array of shape (p, n) - along axis 0 are the list of patterns
        """
        p = self.p
        xi = []
        xiN = []
        for _ in range(p):
            xi_ = self.pattern()
            xiN_ = self.int_rep(xi_)
            while True:
                if xiN_ not in xiN:
                    break
                xi_ = self.pattern()
                xiN_ = self.int_rep(xi_)
            xi.append(xi_)
            xiN.append(xiN_)
        return np.array(xi), xiN

    def C_ij(self, i: int, j: int, XiM: np.ndarray):
        """
        Calculates the strength of coupling from oscillator j to i, according
        to Hebb's learning rule

        Parameters
        ----------
        i, j : int
            Oscillator indices
        XiM : np.ndarray
            The (p, n) matrix of memory patterns

        Returns
        -------
        C_ij value as a float
        """
        p, n = XiM.shape
        cij = 0
        for mu in range(p):
            cij += XiM[mu][i]*XiM[mu][j]
        return cij/n

    def Ct_ij(self, mu: int, nu: int, XiM: np.ndarray):
        """
        Calculates the mu-nu-th element of the (p, p) correlation matrix of the
        memory patterns

        Parameters
        ----------
        mu, nu : int
            Matrix indices
        XiM : np.ndarray
            The (p, n) matrix of memory patterns

        Returns
        -------
        Ct_ij value as a float
        """
        n = self.n
        ctij = 0
        for i in range(n):
            ctij += XiM[mu, i]*XiM[nu, i]
        return ctij/n

    def init_patterns(self):
        """
        Generates the memory patterns and stores them in the attributes.
        """
        XiM, XiMN = self.Xi()
        self.XiM = XiM
        self.XiMN = XiMN

    def init_coupling(self):
        """
        Generates the coupling and correlation matrices and stores them in the
        attributes.
        """
        n = self.n
        p = self.p
        if self.XiM is None:
            self.init_patterns()
        XiM = self.XiM
        C = np.zeros((n, n))
        Ct = np.zeros((p, p))
        for i in range(n):
            for j in range(n):
                C[i, j] = self.C_ij(i, j, XiM)
        for mu in range(p):
            for nu in range(p):
                Ct[mu, nu] = self.Ct_ij(mu, nu, XiM)
        self.C = C
        self.Ct = Ct

    def J_ij(self, i: int, j: int, XiM: np.ndarray, eta: np.ndarray):
        """
        Calculates the ij-th element of the Jacobian* for input pattern eta

        Parameters
        ----------
        i,j : int
            Indices of the matrix
        XiM : np.ndarray
            Memory pattern matrix of shape (p, n)
        eta : np.ndarray
            Input pattern of shape (n, )

        Returns
        -------
        float value of ij-th element of J
        """
        n = self.n
        p = self.p
        jij = 0
        for mu in range(p):
            jij += XiM[mu, i]*XiM[mu, j]*eta[i]*eta[j]
        if i == j:
            for k in range(n):
                for mu in range(p):
                    jij -= XiM[mu, i]*XiM[mu, k]*eta[i]*eta[k]
        return jij/n

    def Jacobian(self, eta: np.ndarray):
        """
        Generates the Jacobian* for stability analysis

        Parameters
        ----------
        eta : np.ndarray
            Input pattern

        Returns
        -------
        Jacobian matrix as a (n, n) numpy array
        """
        n = self.n
        J = np.zeros((n, n))
        if self.XiM is None:
            self.init_patterns()
        XiM = self.XiM
        for _ in range(n**2):
            i = _ // n
            j = _  % n
            J[i, j] += self.J_ij(i, j, XiM, eta)
        return J

class OAMNStabilityAnalysis:
    """
    Class for conducting stability analysis of OAMNs

    Attributes
    ----------
    PBAR : bool
        If progress bar should be displayed or not
    n : int
        Number of oscillators
    p : int
        Number of memory patterns
    eps : float
        Strength of second fourier mode
    n_iter : int
        Number of iterations to consider for analysis

    """
    PBAR = True
    def __init__(self, n: int, p: int, eps: float, n_iter: int):
        """
        Parameters
        ----------
        n : int
            Number of oscillators
        p : int
            Number of memory patterns
        eps : float
            Strength of second fourier mode
        n_iter : int
            Number of iterations to consider for analysis
        """
        self.n = n
        self.p = p
        self.eps = eps
        self.n_iter = n_iter

    def mem_pattern(self, n_iter: int = None):
        """
        Analyse the maximum eigenvalue of Jacobian for memory patterns

        Parameters
        ----------
        n_iter : int
            Number of iterations for the analysis

        Returns
        -------
        list of lambda_max(J) of length n_iter
        """
        n = self.n
        p = self.p
        eps = self.eps
        o = OAMN(n, p, eps, init=False)
        if n_iter is None:
            n_iter = self.n_iter
        lm = []
        if self.PBAR:
            range_t = trange
        else:
            range_t = range
        for _ in range_t(n_iter):
            o.init_patterns()
            eta = o.XiM[np.random.randint(0, p)]
            J = o.Jacobian(eta)
            lm.append(
                eigsh(J, 1, return_eigenvectors=False, which='LA')[0]
            )
        return lm

    def rnd_pattern(self, n_iter: int = None):
        """
        Analyse the maximum eigenvalue of Jacobian for random patterns

        Parameters
        ----------
        n_iter : int
            Number of iterations for the analysis

        Returns
        -------
        list of lambda_max(J) of length n_iter
        """
        n = self.n
        p = self.p
        eps = self.eps
        o = OAMN(n, p, eps)
        if n_iter is None:
            n_iter = self.n_iter
        lm = []
        if self.PBAR:
            range_t = trange
        else:
            range_t = range
        for _ in range_t(n_iter):
            eta = o.nonmem_pattern()
            J = o.Jacobian(eta)
            lm.append(
                eigsh(J, 1, return_eigenvectors=False, which='LA')[0]
            )
        return lm

    def smx_pattern(self, s: int = 3, n_iter: int = None):
        """
        Analyse the maximum eigenvalue of Jacobian for s-mixture patterns

        Parameters
        ----------
        s : int
            Number of memory patterns to mix (default: 3)
        n_iter : int
            Number of iterations for the analysis

        Returns
        -------
        list of lambda_max(J) of length n_iter
        """
        n = self.n
        p = self.p
        eps = self.eps
        o = OAMN(n, p, eps, init=False)
        if n_iter is None:
            n_iter = self.n_iter
        lm = []
        if self.PBAR:
            range_t = trange
        else:
            range_t = range
        for _ in range_t(n_iter):
            o.init_patterns()
            eta = o.smixture_pattern(s)
            J = o.Jacobian(eta)
            lm.append(
                eigsh(J, 1, return_eigenvectors=False, which='LA')[0]
            )
        return lm

    def kbe_pattern(self, k: int = 1, n_iter: int = None):
        """
        Analyse the maximum eigenvalue of Jacobian for k-bit error patterns

        Parameters
        ----------
        k : int
            Number of errors in the memory pattern
        n_iter : int
            Number of iterations for the analysis

        Returns
        -------
        list of lambda_max(J) of length n_iter
        """
        n = self.n
        p = self.p
        eps = self.eps
        o = OAMN(n, p, eps, init=False)
        if n_iter is None:
            n_iter = self.n_iter
        lm = []
        if self.PBAR:
            range_t = trange
        else:
            range_t = range
        for _ in range_t(n_iter):
            o.init_patterns()
            eta = o.XiM[np.random.randint(0, p)]
            eta = o.kbiterror_pattern(eta, k)
            J = o.Jacobian(eta)
            lm.append(
                eigsh(J, 1, return_eigenvectors=False, which='LA')[0]
            )
        return lm


if __name__ == "__main__":
    pass
