"""
In this assignment you should find the intersection points for two functions.
"""
# import matplotlib as plt
import numpy as np
import time
from collections.abc import Iterable
# import matplotlib.pyplot as plt


class Assignment2:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass

    def intersections(self,f1: callable, f2: callable, a: float, b: float, maxerr=0.001) -> Iterable:
        """
        searches for intersections between the two functions f1 and f2 in the given domain [a,b].
        :param f1: function
        :param f2: function
        :param maxerr: the error tolerance
        :return:
        """


        suspect = []

        f_t = lambda x: f1(x) - f2(x)  # define the function for intersection (we will find its roots)
        maxinter = 500

        if abs(f_t(a)) <= maxerr:
            suspect.append(a)
            maxinter -= 1
        if abs(f_t(b)) < maxerr:
            suspect.append(b)
            maxinter -= 1
        partitions = list(np.linspace(a, b, maxinter + 1))

        for i in range(1, len(partitions)):

            if f_t(partitions[i - 1]) * f_t(partitions[i]) < 0:
                suspect.append(self.bisection(partitions[i - 1], partitions[i], f_t, maxerr))



        return suspect




    def bisection(self, left: float, right: float, ft: np.poly1d, err: int = 0.001):  # (b-a) *  maxerr ===> dleta
        """
        standard bisection by the book
        """
        mid = (right + left) * 0.5
        while (abs(right - left) >  err) or (abs(ft(mid)) > err): #check if the error is in range of both x and y
            mid = 0.5 * (right + left)
            if (ft(left) * ft(mid)) < 0:
                right = mid
            else:
                left = mid
        return mid






##########################################################################

import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment2(unittest.TestCase):

    def test_sqr(self):

        ass2 = Assignment2()

        f1 = np.poly1d([-1, 0, 1])
        f2 = np.poly1d([1, 0, -1])

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

    def test_poly(self):

        ass2 = Assignment2()

        f1, f2 = randomIntersectingPolynomials(10)

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))


if __name__ == "__main__":
    unittest.main()