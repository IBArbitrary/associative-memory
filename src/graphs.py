"""
Tools from/for graph theory.
"""

# imports
# import matplotlib.pyplot as plt
import numpy as np
# from scipy.sparse.linalg import eigsh
# from tqdm import trange

class GSET:
    """
    Implementation of GSET notation, from MAX-CUT benchmark dataset

    Attributes
    ----------
    loc : str
        Location of the GSet file
    gstr : str
        The GSet definition as a string
    A : np.ndarray
        The adjaceny matrix of the graph
    """

    def __init__(self, loc: str):
        """
        Parameters
        ----------
        loc : str
            The location of the GSet text file
        """
        self.loc = loc
        self.gstr = ""
        with open(loc, encoding='utf-8') as file:
            self.gstr = file.read()
        self.A = None

    def intize(self, lst: list):
        """
        Makes a list of integers in string representation into ints

        Parameters
        ----------
        lst : list
            List of str(int)

        Returns
        -------
        List of integers
        """
        return [int(_) for _ in lst]

    def parser(self):
        """
        Parses and stores the (weighted) adjacency matrix of the given GSet

        Returns
        -------
        The adjaceny matrix as a np.ndarray
        """
        lines = self.gstr.split("\n")
        N, E = self.intize(lines[0].split())
        edges = lines[1:]
        assert E == len(edges)
        A = np.zeros((N, N))
        for _ in range(E):
            edge = edges[_].split()
            i, j, wij = self.intize(edge)
            A[i-1, j-1] = wij
        A = A + A.T
        self.A = A
        return A

if __name__ == "__main__":
    pass