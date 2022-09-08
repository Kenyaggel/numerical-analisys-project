"""
In this assignment you should interpolate the given function.
"""

import numpy as np
import time
import random

from timeit import default_timer as timer


class Assignment1:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        starting to interpolate arbitrary functions.
        """

        pass

    def interpolate(self, f: callable, a: float, b: float, n: int) -> callable:
        """
        Interpolate the function f in the closed range [a,b] using at most n
        points. Your main objective is minimizing the interpolation error.
        Your secondary objective is minimizing the running time.
        The assignment will be tested on variety of different functions with
        large n values.

        Interpolation error will be measured as the average absolute error at
        2*n random points between a and b. See test_with_poly() below.

        Note: It is forbidden to call f more than n times.

        Note: This assignment can be solved trivially with running time O(n^2)
        or it can be solved with running time of O(n) with some preprocessing.
        **Accurate O(n) solutions will receive higher grades.**

        Note: sometimes you can get very accurate solutions with only few points,
        significantly less than n.

        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        n : int
            maximal number of points to use.

        Returns
        -------
        The interpolating function.
        """

        # replace this line with your solution to pass the second test
        if n == 1:
            result = lambda x: x;

        else:
            def Thomas(low, mid, high, sol):
                """
                implementation of the tridiagonal matrix algorithm, also known as the Thomas algorithm for the solution
                of tridiagonal systems of equations in linear time.
                """
                deg = len(sol)
                low_copy, mid_copy, high_copy, sol_copy = map(np.array, (low, mid, high, sol))
                for i in range(1, deg):
                    coef = low_copy[i - 1] / mid_copy[i - 1]
                    mid_copy[i] = mid_copy[i] - coef * high_copy[i - 1]
                    sol_copy[i] = sol_copy[i] - coef * sol_copy[i - 1]

                x_vec = mid_copy
                x_vec[-1] = sol_copy[-1] / mid_copy[-1]

                for i in range(deg - 2, -1, -1):
                    x_vec[i] = (sol_copy[i] - high_copy[i] * x_vec[i + 1]) / mid_copy[i]

                return x_vec

            def get_cubic(a, b, c, d):
                """
                :param a,b,c,d: points used for the cubic bezier curve.
                :return: a function that represents the cubic bezier curve.
                """
                return lambda t: np.power(1 - t, 3) * a + 3 * np.power(1 - t, 2) * t * b + 3 * (1 - t) * \
                                 np.power(t, 2) * c + np.power(t, 3) * d

            def get_bezier_coef(x, y):
                """
                calculates control points using the given
                :param x: list of the x coordinates of the points from the source function.
                :param y: list of the y coordinates of the points from the source function.
                :return: A_cont,B_cont: the calculated control points for the points given by the function.
                """
                n = len(x) - 1

                high = np.full(n - 1, 1.)
                mid = np.full(n, 4.)
                mid[-1] = 7.
                mid[0] = 2.
                low = np.full(n - 1, 1.)
                low[-1] = 2.

                # to claculate the controll points we need to solve a tridiagonal system of equation. the non zero
                # coefficients of the system (every list represents the main upper and lower diagonals of the equation)
                Py = [2 * (2 * y[i] + y[i + 1]) for i in range(n)]
                Py[0] = y[0] + 2 * y[1]
                Py[n - 1] = 8 * y[n - 1] + y[n]

                # solve the equation using thomas algorithm to get the first list of control points.
                A_cont = Thomas(low, mid, high, Py)
                B_cont = np.empty(n)

                # the second list of control points are given by the first controll point and the second real point for
                # every curve
                for i in range(n - 1):
                    B_cont[i] = 2 * y[i + 1] - A_cont[i + 1]

                B_cont[n - 1] = (A_cont[n - 1] + y[n]) / 2

                return A_cont, B_cont

            def get_bezier_cubic(x, y):
                """
                runs the get cubic for all the points and control points in and reruns a list of itnerpolated functions.
                """
                A_vec, B_vec = get_bezier_coef(x, y)
                return [
                    get_cubic(y[i], A_vec[i], B_vec[i], y[i + 1])
                    for i in range(len(x) - 1)
                ]

            ran = np.linspace(a, b, n)
            y = [f(x) for x in ran]
            inter_y = get_bezier_cubic(ran, y)

            def g(x_val):
                for i in range(n - 1):
                    if ran[i] <= x_val <= ran[i + 1]:
                        est = (x_val - ran[i]) / (ran[i + 1] - ran[i])

                        return inter_y[i](est)

        return g


##########################################################################


import unittest
from functionUtils import *
from tqdm import tqdm


class TestAssignment1(unittest.TestCase):

    def test_with_poly(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0

        d = 30
        for i in tqdm(range(1)):  # changed from 100 to 1
            a = np.random.randn(d)

            f = np.poly1d(a)

            ff = ass1.interpolate(f, -10, 10, 100)

            xs = np.random.random(200)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / 200
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print(T)
        print(mean_err)

    def test_with_poly_restrict(self):
        ass1 = Assignment1()
        a = np.random.randn(5)
        f = RESTRICT_INVOCATIONS(10)(np.poly1d(a))
        ff = ass1.interpolate(f, -10, 10, 10)
        xs = np.random.random(20)
        for x in xs:
            yy = ff(x)


if __name__ == "__main__":
    unittest.main()
