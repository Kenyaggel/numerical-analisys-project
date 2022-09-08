"""
In this assignment you should fit a model function of your choice to data 
that you sample from a contour of given shape. Then you should calculate
the area of that shape. 

The sampled data is very noisy so you should minimize the mean least squares 
between the model you fit and the data points you sample.  

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You 
must make sure that the fitting function returns at most 5 seconds after the 
allowed running time elapses. If you know that your iterations may take more 
than 1-2 seconds break out of any optimization loops you have ahead of time.

Note: You are allowed to use any numeric optimization libraries and tools you want
for solving this assignment. 
Note: !!!Despite previous note, using reflection to check for the parameters 
of the sampled function is considered cheating!!! You are only allowed to 
get (x,y) points from the given shape by calling sample(). 
"""

import numpy as np
import time
import random
from functionUtils import AbstractShape
import matplotlib.pyplot as plt

class MyShape(AbstractShape):
    # change this class with anything you need to implement the shape
    def __init__(self, points):
        self.points = points



    def area(self):


        S = 0
        # calculate the area by adding and subtracting trapezoids.
        for i in range(len(self.points) - 1):
            S += (self.points[i + 1][0] - self.points[i][0]) * 0.5 * (
                        self.points[i][1] + self.points[i + 1][1])
        return abs(np.float32(S))

class Assignment5:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass

    def area(self, contour: callable, maxerr=0.001)->np.float32:
        """
        Compute the area of the shape with the given contour. 

        Parameters
        ----------
        contour : callable
            Same as AbstractShape.contour 
        maxerr : TYPE, optional
            The target error of the area computation. The default is 0.001.

        Returns
        -------
        The area of the shape.

        """
        num_samp = int(0.125/maxerr)
        points = np.array(contour(num_samp))


        S = 0
        for i in np.arange(len(points)-1):
            # there will be error if the two points cross the axis.
            S +=  (points[i+1][0] - points[i][0])*0.5 * (points[i][1] + points[i+1][1])

        return abs(np.float32(S))


    def fit_shape(self, sample: callable, maxtime: float) -> AbstractShape:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape. 
        
        Parameters
        ----------
        sample : callable. 
            An iterable which returns a data point that is near the shape contour.
        maxtime : float
            This function returns after at most maxtime seconds. 

        Returns
        -------
        An object extending AbstractShape. 
        """
        num_of_sample = maxtime * 5000


        samp = [sample() for i in np.arange(num_of_sample)]
        #make referance point

        ref = np.average([s[0] for s in samp]),np.average([s[1] for s in samp])

        def sort_key(point):
            """the key calculates the angle that x,y creates with ref"""
            if np.abs(point[0] - ref[0]) <= 0.000001:
                tg = np.inf
            else:
                tg = (point[1] - ref[1]) / (point[0] - ref[0])
            ang = np.arctan(tg)
            if point[0] - ref[0] < 0:
                if ang >= 0:
                    return ang - (np.pi)
                return ang + (np.pi)
            return ang

        points =[]
        sorted_sump = sorted(samp, key = sort_key)
        # if i were to use all the points i would get a shape that overfits because of the noise, so i use 10 points
        # to further reduce the Gaussian noise.
        step = int(np.ceil(len(sorted_sump)/500))
        for i in np.arange(step,len(sorted_sump),step):

            points.append((np.mean([x[0] for x in sorted_sump[i-step:i]]),np.mean([y[1] for y in sorted_sump[i-step:i]])))

        return MyShape(points)





#
# #########################################################################
#
#
#
# import unittest
# from sampleFunctions import *
# from tqdm import tqdm
#
#
# class TestAssignment5(unittest.TestCase):
#
#     def test_return(self):
#         circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
#         ass5 = Assignment5()
#         T = time.time()
#         shape = ass5.fit_shape(sample=circ, maxtime=5)
#         T = time.time() - T
#         self.assertTrue(isinstance(shape, AbstractShape))
#         self.assertLessEqual(T, 5)
#
#     def test_delay(self):
#         circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
#
#         def sample():
#             return circ()
#
#         ass5 = Assignment5()
#         T = time.time()
#         shape = ass5.fit_shape(sample=sample, maxtime=5)
#         T = time.time() - T
#         self.assertTrue(isinstance(shape, AbstractShape))
#         self.assertGreaterEqual(T, 5)
#
#     def test_circle_area(self):
#         circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
#         ass5 = Assignment5()
#         T = time.time()
#         shape = ass5.fit_shape(sample=circ, maxtime=30)
#         T = time.time() - T
#         a = shape.area()
#         self.assertLess(abs(a - np.pi), 0.01)
#         self.assertLessEqual(T, 32)
#
#     def test_bezier_fit(self):
#         circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
#         ass5 = Assignment5()
#         T = time.time()
#         shape = ass5.fit_shape(sample=circ, maxtime=30)
#         T = time.time() - T
#         a = shape.area()
#         self.assertLess(abs(a - np.pi), 0.01)
#         self.assertLessEqual(T, 32)
#
#
# if __name__ == "__main__":
#     unittest.main()
