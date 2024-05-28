"""
Oscillatory Associative Memory Network (OAMN) module.
Based on Nishikawa et al. (2004) (doi: 10.1016/j.physd.2004.06.011)
"""

# imports
# import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import eigsh
from tqdm import trange
from .graphs import GSET

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

    def EnergyFunction(self, C: np.ndarray, eta: np.ndarray, eps: float):
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

    def EulerSolver(
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
        thetas = np.zeros((T+1, n))
        thetas[0] = th0
        for t in range(T):
            th_ = thetas[t]
            thetas[t+1] = th_ + dt*self.dtheta(th_, C, eps)
        return thetas

    def ConvEulerSolver(
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
        thetas = [th0]
        t = 0
        while True:
            th_ = thetas[t]
            th = th_ + dt*self.dtheta(th_, C, eps)
            thetas.append(th)
            t += 1
            if np.mean(np.abs(th - th_)) < tol:
                break
        tc = t
        for _ in range(exs):
            th_ = thetas[tc]
            th = th_ + dt*self.dtheta(th_, C, eps)
            thetas.append(th)
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
                dth = np.round((theta[j] - theta[i])/np.pi)
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

class MaxCutProblem:
    """
    Class for solving the maximum cut problem using oscillatory associative
    memory networks

    Attributes
    ----------
    loc : str
        Location of the GSet file for the graph
    gset : graphs.GSET
        GSET object for the graph
    G : np.ndarray
        Adjacency matrix of the graph
    N : int
        Number of vertices/oscillators
    """

    PBAR = True

    def __init__(self, loc: str):
        """
        Parameters
        ----------
        loc : str
            Location of the GSet file
        """
        self.loc = loc
        self.gset = GSET(loc)
        self.G = self.gset.parser()
        self.n = self.gset.N
        self.eps = None
        self.dt = None
        self.t_ = None
        self.solver = None
        self.o = None
        self.n_sample = None
        self.etas = []
        self.engs = []
        self.emax = None
        self.cuts = None
        self.vcuts = []

    def system(self, eps: float, dt: float, T: float, n_sample: float):
        """
        Define the oscillator system and integration parameters for solving
        the max-cut problem

        Parameters
        ----------
        eps : float
            Strength of the second Fourier mode
        dt : float
            Time interval for the integration
        T : float
            If greater than 1, number of iterations, if less than 1 tolerance
        n_sample : int
            Sample size to simulate
        """
        self.dt = dt
        self.eps = eps
        self.n_sample = n_sample
        self.o = OAMN(self.n, 3, eps, init = False)
        if T >= 1:
            self.solver = self.o.EulerSolver
            T = int(T)
        elif 0 < T < 1:
            self.solver = self.o.ConvEulerSolver
        self.t_ = T

    def solve(self):
        """
        Solves the maximum cut problem and gives the
        """
        if self.PBAR:
            range_t = trange
        else:
            range_t = range
        for _ in range_t(self.n_sample):
            th_ = self.o.rnd_theta()
            mc_ = self.solver(
                th_, self.G, self.dt, self.eps, self.t_, exs = 10, sil = True
            )
            et_ = self.o.construct_pattern(mc_[-1])
            en_ = self.o.EnergyFunction(self.G, et_, self.eps)
            self.etas.append(et_)
            self.engs.append(en_)
        self.emax = max(set(self.engs))
        cuts = []
        for _ in range(self.n_sample):
            en = self.engs[_]
            if en == self.emax:
                cuts.append(self.o.int_rep(self.etas[_]))
        cuts = np.array([
            self.o.bin_rep(_, self.n) for _ in list(set(cuts))
        ])
        self.cuts = cuts
        for cut in cuts:
            part1, part2 = set(), set()
            for _, vi in enumerate(cut):
                if vi == 1:
                    part2.add(_+1)
                else:
                    part1.add(_+1)
            self.vcuts.append([part1, part2])
        return self.vcuts


if __name__ == "__main__":
    pass
