"""
Self-Consistent Signal-to-Noise Analysis (SCSNA) tools for studying the
equilibrium properties of Oscillator Neural Networks (ONNs).
Based on Aonishi (1998) (doi: 10.1103/PhysRevE.58.4865)
"""

from dataclasses import dataclass
from typing import Callable
import numpy as np
# from tqdm import trange
from scipy.integrate import dblquad
from scipy.optimize import fsolve

@dataclass
class Params:
    """
    Parameter dataclass to be used in conjunction with SCSNA class

    Attributes
    ----------
    Q : np.ndarray
        The covariance matrix
    L : float
        Lambda
    alpha : float
        Capacity p/n of the oscillator network
    mc : float
        Order parameter
    ms : float
        Overlap order parameter
    qs : float
        Not exactly sure what this is
    qc : float
        Not exactly sure what this is
    qsc : float
        Not exactly sure what this is
    c1 : float
        Coefficient of expansion
    c2 : float
        Coefficient of expansion
    s1 : float
        Coefficient of expansion
    s2 : float
        Coefficient of expansion
    """
    Q: np.ndarray
    alpha: float
    L: float
    mc: float
    ms: float
    qs: float
    qc: float
    qsc: float
    c1: float
    c2: float
    s1: float
    s2: float
    est : np.ndarray

class SCSNA:
    """
    Class for conducting Self-Consistent Signal-to-Noise Analysis on the
    Oscillator Neural Network System.

    Attributes
    ----------
    params : Params
        The parameters object
    """

    def __init__(self, params: Params):
        """
        Parameters
        ----------
        params : Params
            Parameters for the SCSNA
        """
        self.params = params

    def update_est(self, est_: np.ndarray):
        """
        Changes the estimate for solving the equilibrium condition equation

        Parameters
        ----------
        est_ : np.ndarray
            The new estimate
        """
        self.params.est = est_

    def update_params(self, params_ : Params):
        """
        Changes the parameters of the system

        Parameters
        ----------
        params_ : Params
            The new parameters object
        """
        self.params = params_

    def gaussian_average(self, f: Callable, p: Params, alt: bool = False):
        """
        Calculates the gaussian average of a given function defined on R^2

        Parameters
        ----------
        f : Callable
            The function to average
        p : Params
            Parameters for the function and averaging
        alt : Bool
            Determines method of calculating Y

        Returns
        -------
        tuple of the gaussian average (float) and error (float)
        """
        detQ = np.linalg.det(p.Q)
        invQ = np.linalg.inv(p.Q)
        def kernel(x1: float, x2: float):
            X = np.array([x1, x2])[np.newaxis]
            return (1/(2*np.pi*np.sqrt(detQ)) * \
                np.exp((-1/2)*(X @ invQ @ X.T)))[0, 0]
        def integrand(x1: float, x2: float, p: Params, alt: bool):
            return f(x1, x2, p, alt)*kernel(x1, x2)
        return dblquad(
            integrand, -np.inf, np.inf, -np.inf, np.inf,
            args=(p, alt)
        )

    @staticmethod
    def q_mat(q1: float, q3: float, q2: float) -> np.ndarray:
        """
        Creates the covariance matrix using its elements

        Parameters
        ----------
        q1, q2, q3 : float
            first, second/third, and fourth element of the matrix

        Returns
        -------
        (2, 2) np.ndarray of the covariance matrix
        """
        return np.array([
            [q1, q3],
            [q3, q2]
        ])

    def x(
        self, x1: float, x2: float, p: Params,
        *args, alt: bool = False
        ) -> float:
        """
        Calculates cos(phi)

        Parameters
        ----------
        x1, x2 : float
            Coordinates
        p : Params
            Parameters object
        alt : Bool
            Determines method of calculating X

        Returns
        -------
        float value of cos(phi)
        """
        del p, args
        if alt:
            return self.x_alt(x1, x2, p)
        return x1/np.sqrt(x1**2 + x2**2)

    def y(
        self, x1: float, x2: float, p: Params,
        *args, alt: bool = False
        ) -> float:
        """
        Calculates sin(phi)

        Parameters
        ----------
        x1, x2 : float
            Coordinates
        p : Params
            Parameters object
        alt : Bool
            Determines method of calculating Y

        Returns
        -------
        float value of sin(phi)
        """
        del p, args
        if alt:
            return self.y_alt(x1, x2, p)
        return x2/np.sqrt(x1**2 + x2**2)

    def xs(
        self, x1: float, x2: float, p: Params,
        *args, alt: bool = False
        ) -> float:
        """
        Calculates (cos(phi))^2

        Parameters
        ----------
        x1, x2 : float
            Coordinates
        p : Params
            Parameters object
        alt : Bool
            Determines method of calculating X/Y

        Returns
        -------
        float value of (cos(phi))^2
        """
        x_ = self.x(x1, x2, p, alt=alt, *args)
        return x_**2

    def ys(
        self, x1: float, x2: float, p: Params,
        *args, alt: bool = False
        ) -> float:
        """
        Calculates (sin(phi))^2

        Parameters
        ----------
        x1, x2 : float
            Coordinates
        p : Params
            Parameters object
        alt : Bool
            Determines method of calculating X/Y

        Returns
        -------
        float value of (sin(phi))^2
        """
        y_ = self.y(x1, x2, p, alt=alt, *args)
        return y_**2

    def xy(
        self, x1: float, x2: float, p: Params,
        *args, alt: bool = False
        ) -> float:
        """
        Calculates sin(phi)*cos(phi)

        Parameters
        ----------
        x1, x2 : float
            Coordinates
        p : Params
            Parameters object
        alt : Bool
            Determines method of calculating X/Y

        Returns
        -------
        float value of (sin(phi))*(cos(phi))
        """
        x_ = self.x(x1, x2, p, alt=alt, *args)
        y_ = self.y(x1, x2, p, alt=alt, *args)
        return x_*y_

    def mc(self, alt: bool = False):
        """
        Calculates the gaussian average of cos(phi) using the x function

        Parameters
        ----------
        alt : Bool
            Determines method of calculating X/Y

        Returns
        -------
        float value of the gaussian average
        """
        return self.gaussian_average(self.x, self.params, alt=alt)

    def ms(self, alt: bool = False):
        """
        Calculates the gaussian average of sin(phi) using the y function

        Parameters
        ----------
        alt : Bool
            Determines method of calculating X/Y

        Returns
        -------
        float value of the gaussian average
        """
        return self.gaussian_average(self.y, self.params, alt=alt)

    def qc(self, alt: bool = False):
        """
        Calculates the gaussian average of (cos(phi))^2 using the x function

        Parameters
        ----------
        alt : Bool
            Determines method of calculating X/Y

        Returns
        -------
        float value of the gaussian average
        """
        return self.gaussian_average(self.xs, self.params, alt=alt)

    def qs(self, alt: bool = False):
        """
        Calculates the gaussian average of (sin(phi))^2 using the y function

        Parameters
        ----------
        alt : Bool
            Determines method of calculating X/Y

        Returns
        -------
        float value of the gaussian average
        """
        return self.gaussian_average(self.ys, self.params, alt=alt)

    def qsc(self, alt: bool = False):
        """
        Calculates the gaussian average of (sin(phi)*cos(phi)) using the x and
        y functions

        Parameters
        ----------
        alt : Bool
            Determines method of calculating X/Y

        Returns
        -------
        float value of the gaussian average
        """
        return self.gaussian_average(self.xy, self.params, alt=alt)

    def csikern(
        self, x1: float, x2: float, p: Params,
        cs: str, i: int, alt: bool = False
        ) -> float:
        """
        Calculates the kernel for cofficients C1/C2/S1/S2 calculation

        Parameters
        ----------
        x1, x2 : float
            Coordinates
        p : Params
            Parameters object
        cs : int
            Function to consider
        i : int
            Index of the coefficient (takes values 1 and 2)
        alt : Bool
            Determines method of calculating X/Y


        Returns
        -------
        float value of the kernel
        """
        match cs:
            case 'c':
                xy = self.x
            case 's':
                xy = self.y
            case _:
                raise ValueError("Invalid choice of function")
        p = self.params
        qv = p.Q[:, i-1]
        x0 = np.array([x1, x2])[np.newaxis]
        return (qv @ x0.T)*xy(x1, x2, p, alt=alt)

    def c1(self, alt: bool = False) -> float:
        """
        Calculates the C1 coefficient of expansion

        Parameters
        ----------
        alt : Bool
            Determines method of calculating X/Y

        Returns
        -------
        float value of the coefficient
        """
        p = self.params
        def kern(x1, x2, p, alt):
            return self.csikern(x1, x2, p, 'c', 1, alt=alt)
        return ((1-p.L)/np.sqrt(p.alpha)) * \
                (self.gaussian_average(kern, p, alt=alt)[0])

    def c2(self, alt: bool = False) -> float:
        """
        Calculates the C2 coefficient of expansion

        Parameters
        ----------
        alt : Bool
            Determines method of calculating X/Y

        Returns
        -------
        float value of the coefficient
        """
        p = self.params
        def kern(x1, x2, p, alt):
            return self.csikern(x1, x2, p, 'c', 2, alt=alt)
        return ((1-p.L)/np.sqrt(p.alpha)) * \
                (self.gaussian_average(kern, p, alt=alt)[0])

    def s1(self, alt: bool = False) -> float:
        """
        Calculates the S1 coefficient of expansion

        Parameters
        ----------
        alt : Bool
            Determines method of calculating X/Y

        Returns
        -------
        float value of the coefficient
        """
        p = self.params
        def kern(x1, x2, p, alt):
            return self.csikern(x1, x2, p, 's', 1, alt=alt)
        return ((1-p.L)/np.sqrt(p.alpha)) * \
                (self.gaussian_average(kern, p, alt=alt)[0])

    def s2(self, alt: bool = False) -> float:
        """
        Calculates the S2 coefficient of expansion

        Parameters
        ----------
        alt : Bool
            Determines method of calculating X/Y

        Returns
        -------
        float value of the coefficient
        """
        p = self.params
        def kern(x1, x2, p, alt):
            return self.csikern(x1, x2, p, 's', 2, alt=alt)
        return ((1-p.L)/np.sqrt(p.alpha)) * \
                (self.gaussian_average(kern, p, alt=alt)[0])

    def equilibrium_cond(
            self, phi: float, x1: float, x2: float, p: Params
        ) -> float:
        """
        Calculates the value of the equilbrium condition equation for a given
        value of phi

        Parameters
        ----------
        phi : float
            The angle phi
        x1, x2 : float
            Coordinates
        p : Params
            Parameters object

        Returns
        -------
        float value of the equation
        """
        sphi = np.sin(phi)
        cphi = np.cos(phi)
        alpha2 = np.sqrt(p.alpha)
        return p.alpha*(p.c2*(sphi**2) - p.s1*(cphi**2) \
                + (p.c1 - p.c2)*sphi*cphi) \
                + sphi*((1-p.L)*p.mc + alpha2*x1) \
                - cphi*((1-p.L)*p.ms + alpha2*x2)

    def phi(
        self, x1: float, x2: float, p: Params, full: bool = False
        ):
        """
        Calculates the root values phi numerically

        Parameters
        ----------
        x1, x2 : float
            Coordinates
        p : Params
            The parameters object

        Returns
        -------
        list of phi which solves the equilibrium condition
        """
        est = self.params.est
        out = fsolve(
            self.equilibrium_cond, est,
            args = (x1, x2, p)
        )
        out = np.unique(np.round(out, 8))
        out = out[out >= 0]
        o0 = out[0]
        out = out[out <= o0 + 2*np.pi]
        if full:
            return out
        return out[-1]

    def x_alt(self, x1: float, x2: float, p: Params) -> float:
        """
        Calculates the value of cos(phi) numerically

        Parameters
        ----------
        x1, x2 : float
            Coordinaes
        p : Params
            The parameters object

        Returns
        -------
        float value of cos(phi)
        """
        phi = self.phi(x1, x2, p)
        return np.cos(phi)

    def y_alt(self, x1: float, x2: float, p: Params) -> float:
        """
        Calculates the value of sin(phi) numerically

        Parameters
        ----------
        x1, x2 : float
            Coordinaes
        p : Params
            The parameters object

        Returns
        -------
        float value of sin(phi)
        """
        phi = self.phi(x1, x2, p)
        return np.sin(phi)


if __name__ == "__main__":
    # Q = SCSNA.q_mat(0.9, 0.5, 0.5)
    # p1 = Params(
    #     Q = Q, L = 0.6, alpha = 0.01, mc = 0.2, ms = 0.96,
    #     c1 = 0.6, c2 = 0, s1 = 0, s2 = 1, qc = 1, qs = 0, qsc = 0,
    #     est = np.linspace(0, 2*np.pi, 100)
    # )
    # sna1 = SCSNA(p1)
    # # sna1.gaussian_average(f = sna1.x_alt, p = sna1.params)
    # sna1.c1()
    pass
