"""
Maximum cut problem, solved using oscillatory associative memory networks.
"""

import numpy as np
from tqdm import trange
from .graphs import GSET
from .oamn import OAMN

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
    n : int
        Number of vertices/oscillators
    eps : float
        Strength of the second Fourier mode
    dt : float
        Time interval for numerical integration
    t_ : float/int
        Tolerance/Total time of iteration
    solver : callable
        Type of solver function to use
    o : oamn.OAMN
        OAMN object
    n_sample : int
        Number of samples to iterate over
    etas : list
        List of solution configurations
    engs : list
        List of energies of the solutions
    emax : float
        Maximum energy of the solutions
    cuts : list
        List of solutions as binary patterns
    vcuts : list
        List of solutions as vertices subsets
    mctraj : np.ndarray
        The final converging solution trajectory
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
        self.mctraj = None

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
            self.solver = self.o.euler_solver
            T = int(T)
        elif 0 < T < 1:
            self.solver = self.o.conv_euler_solver
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
                th_, -self.G, self.dt, self.eps, self.t_, exs = 0, sil = True
            )
            self.mctraj = mc_
            et_ = self.o.construct_pattern(mc_[-1])
            en_ = self.o.energy_function(-self.G, et_, self.eps)
            self.etas.append(et_)
            self.engs.append(en_)
        self.emax = min(set(self.engs))
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
