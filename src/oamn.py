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
    xi_mat : np.ndarray
        (p, n) array of memory patterns
    xi_n : list
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
        self.xi_mat = None
        self.xi_n = None
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
        Calculates the integer representation of a binary pattern

        Parameters
        ----------
        pattern : np.ndarray
            The binary pattern

        Returns
        -------
        int value of the integer representation
        """
        return int("".join([str(_) for _ in (pattern+1)//2]), 2)

    def bin_rep(self, intrep: int, n: int):
        """
        Calculates the binary pattern using integer representation

        Parameters
        ----------
        intrep : int
            The integer representation of the pattern
        n : int
            Length of the binary pattern
        Returns
        -------
        np.ndarray of the binary pattern
        """
        return [(2*int(_))-1 for _ in (("{0:b}".format(intrep)).zfill(n))]

    def nonmem_pattern(self):
        """
        Generates a random pattern (exclusive of the memory patterns)

        Returns
        -------
        numpy array of length n for the binary pattern
        """
        p1 = self.pattern()
        p1N = self.int_rep(p1)
        if self.xi_mat is None:
            self.xi()
        while True:
            if p1N not in self.xi_n:
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
        if self.xi_mat is None:
            self.init_patterns()
        xi_mat = self.xi_mat
        eta_s = xi_mat[:s]
        n = self.n
        eta = np.zeros(n)
        for _ in range(s):
            eta += eta_s[_]
        eta = np.array([_//np.abs(_) for _ in eta])
        return eta

    def xi(self):
        """
        Generates the n x p matrix xi of the p-patterns to be memorised,
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

    def C_ij(self, i: int, j: int, xi_mat: np.ndarray):
        """
        Calculates the strength of coupling from oscillator j to i, according
        to Hebb's learning rule

        Parameters
        ----------
        i, j : int
            Oscillator indices
        xi_mat : np.ndarray
            The (p, n) matrix of memory patterns

        Returns
        -------
        C_ij value as a float
        """
        p, n = xi_mat.shape
        cij = 0
        for mu in range(p):
            cij += xi_mat[mu][i]*xi_mat[mu][j]
        return cij/n

    def Ct_ij(self, mu: int, nu: int, xi_mat: np.ndarray):
        """
        Calculates the mu-nu-th element of the (p, p) correlation matrix of the
        memory patterns

        Parameters
        ----------
        mu, nu : int
            Matrix indices
        xi_mat : np.ndarray
            The (p, n) matrix of memory patterns

        Returns
        -------
        Ct_ij value as a float
        """
        n = self.n
        ctij = 0
        for i in range(n):
            ctij += xi_mat[mu, i]*xi_mat[nu, i]
        return ctij/n

    def init_patterns(self):
        """
        Generates the memory patterns and stores them in the attributes.
        """
        xi_mat, xi_n = self.xi()
        self.xi_mat = xi_mat
        self.xi_n = xi_n

    def init_coupling(self):
        """
        Generates the coupling and correlation matrices and stores them in the
        attributes.
        """
        n = self.n
        p = self.p
        if self.xi_mat is None:
            self.init_patterns()
        xi_mat = self.xi_mat
        C = np.zeros((n, n))
        Ct = np.zeros((p, p))
        for i in range(n):
            for j in range(n):
                C[i, j] = self.C_ij(i, j, xi_mat)
        for mu in range(p):
            for nu in range(p):
                Ct[mu, nu] = self.Ct_ij(mu, nu, xi_mat)
        self.C = C
        self.Ct = Ct

    def J_ij(self, i: int, j: int, xi_mat: np.ndarray, eta: np.ndarray):
        """
        Calculates the ij-th element of the jacobian* for input pattern eta

        Parameters
        ----------
        i,j : int
            Indices of the matrix
        xi_mat : np.ndarray
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
            jij += xi_mat[mu, i]*xi_mat[mu, j]*eta[i]*eta[j]
        if i == j:
            for k in range(n):
                for mu in range(p):
                    jij -= xi_mat[mu, i]*xi_mat[mu, k]*eta[i]*eta[k]
        return jij/n

    def jacobian_(self, eta: np.ndarray):
        """
        Generates the jacobian* for stability analysis (legacy)

        Parameters
        ----------
        eta : np.ndarray
            Input pattern

        Returns
        -------
        jacobian matrix as a (n, n) numpy array
        """
        n = self.n
        J = np.zeros((n, n))
        if self.xi_mat is None:
            self.init_patterns()
        xi_mat = self.xi_mat
        for _ in range(n**2):
            i = _ // n
            j = _ % n
            J[i, j] += self.J_ij(i, j, xi_mat, eta)
        return J

    def jacobian(self, eta: np.ndarray):
        """
        Generates the jacobian* for stability analysis (legacy)

        Parameters
        ----------
        eta : np.ndarray
            Input pattern

        Returns
        -------
        jacobian matrix as a (n, n) numpy array
        """
        n = self.n
        if self.xi_mat is None:
            self.init_patterns()
        xi_mat = self.xi_mat
        M = xi_mat * eta
        J = M.T @ M
        J -= np.diag(np.sum(J, axis=0))
        J = J/n
        return J

    def energy_function(self, C: np.ndarray, eta: np.ndarray, eps: float):
        """
        Calculates the Lyuapunov energy function for given input pattern and
        coupling strength matrix

        Parameters
        ----------
        C : np.ndarray
            Coupling strength matrix
        eta : np.ndarray
            Input pattern
        eps : float
            Strength of the second Fourier mode
        Returns
        -------
        float value of the Lyapunov energy
        """
        n = len(eta)
        eta = eta[np.newaxis]
        return (-(eta @ C @ eta.T)/2 - (n*eps)/4)[0, 0]

    def dtheta(self, theta: np.ndarray, C:np.ndarray, eps: float):
        """
        Calculates the derivative of phase vector of the oscillator system

        Parameters
        ----------
        theta : np.ndarray
            (n, ) array of initial phases
        C : np.ndarray
            (n, n) array of coupling strengths between oscillators
        eps : float
            Strength of the second Fourier mode

        Returns
        -------
        (n, ) array of the derivative vector
        """
        n = len(theta)
        dth = np.zeros(n)
        for i in range(n):
            dthi = 0
            for j in range(n):
                dthi += C[i, j]*np.sin(theta[j]-theta[i]) + \
                (eps/n)*np.sin((theta[j]-theta[i])*2)
            dth[i] += dthi
        return dth

    def rnd_theta(self):
        """
        Generates a random array of theta_i (between 0 and 2pi), used for solving the ODEs

        Returns
        -------
        (n, ) array of theta values
        """
        n = self.n
        th = np.random.uniform(size=(n,))*(2*np.pi)
        return th

    def euler_solver(
        self,
        th0: np.ndarray, C: np.ndarray, dt: float, T: int, eps: float, **kwargs
    ):
        """
        Numerically integrates the ODE for n-coupled oscillators using Euler
        method

        Parameters
        ----------
        th0 : np.ndarray
            (n, ) array of the initial phase vector
        C : np.ndarray
            (n, n) array of the couping strengths matrix
        dt : float
            Time interval
        T : int
            Number of time steps to integrate
        eps : float
            Strength of the second Fourier mode

        Returns
        -------
        (T+1, n) array of the time evolved phase vectors
        """
        n = len(th0)
        T = int(T)
        thetas = np.zeros((T+1, n))
        thetas[0] = th0
        for t in range(T):
            th_ = thetas[t]
            thetas[t+1] = th_ + dt*self.dtheta(th_, C, eps)
        return thetas

    def conv_euler_solver(
        self,
        th0: np.ndarray, C: np.ndarray,
        dt: float, tol: float, eps: float, exs: int = 0, sil: bool = False
    ):
        """
        Numerically integrates the ODE for n-coupled oscillators upto
        convergence using Euler method

        Parameters
        ----------
        th0 : np.ndarray
            (n, ) array of the initial phase vector
        C : np.ndarray
            (n, n) array of the couping strengths matrix
        dt : float
            Time interval
        eps : float
            Strength of the second Fourier mode
        tol : float
            Tolerance value which will be considered as convergence
        exs : int
            The extra iterations after convergence

        Returns
        -------
        (x+1, n) array of the time evolved phase vectors
        """
        thetas = np.array([th0,])
        t = 0
        while True:
            if not sil:
                print(f"Iteration {str(t).zfill(3)}", end='\r')
            th_ = thetas[t]
            th = th_ + dt*(self.dtheta(th_, C, eps))
            np.append(thetas, [th], axis=0)
            t += 1
            if np.mean(np.abs(th - th_)) < tol:
                break
        tc = t
        for _ in range(exs):
            th_ = thetas[tc]
            th = th_ + dt*self.dtheta(th_, C, eps)
            np.append(thetas, [th], axis=0)
            tc += 1
        if not sil:
            print(f'Converged in {t} iterations.')
        return thetas

    def construct_pattern(self, theta: np.ndarray):
        """
        Constructs a binary pattern given the phase vector

        Parameters
        ----------
        theta : np.ndarray
            (n, ) array of phase vector

        Returns
        -------
        (n, ) array of binary pattern
        """
        n = len(theta)
        eta = np.ones(n, dtype='int')
        for i in range(n):
            for j in range(n):
                dth = np.round((theta[j] - theta[i])/np.pi, 3)
                if dth == 0:
                    eta[j] = eta[i]
                elif (dth != 0) and dth.is_integer():
                    eta[j] = -1*eta[i]
        return eta

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
        Analyse the maximum eigenvalue of jacobian for memory patterns

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
            eta = o.xi_mat[np.random.randint(0, p)]
            J = o.jacobian(eta)
            lm.append(
                eigsh(J, 1, return_eigenvectors=False, which='LA')[0]
            )
        return lm

    def rnd_pattern(self, n_iter: int = None):
        """
        Analyse the maximum eigenvalue of jacobian for random patterns

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
            J = o.jacobian(eta)
            lm.append(
                eigsh(J, 1, return_eigenvectors=False, which='LA')[0]
            )
        return lm

    def smx_pattern(self, s: int = 3, n_iter: int = None):
        """
        Analyse the maximum eigenvalue of jacobian for s-mixture patterns

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
            J = o.jacobian(eta)
            lm.append(
                eigsh(J, 1, return_eigenvectors=False, which='LA')[0]
            )
        return lm

    def kbe_pattern(self, k: int = 1, n_iter: int = None):
        """
        Analyse the maximum eigenvalue of jacobian for k-bit error patterns

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
            eta = o.xi_mat[np.random.randint(0, p)]
            eta = o.kbiterror_pattern(eta, k)
            J = o.jacobian(eta)
            lm.append(
                eigsh(J, 1, return_eigenvectors=False, which='LA')[0]
            )
        return lm

if __name__ == "__main__":
    pass
