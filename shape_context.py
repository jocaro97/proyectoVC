import numpy as np
import cv2
import math
from scipy.spatial.distance import cdist, cosine
from scipy.optimize import linear_sum_assignment
from lapjv import lapjv
import matplotlib.pyplot as plt

class ShapeContext(object):

    def __init__(self, nbins_r=5, nbins_theta=12, r_inner=0.1250, r_outer=2.0):
        # number of radius zones
        self.nbins_r = nbins_r
        # number of angles zones
        self.nbins_theta = nbins_theta
        # maximum and minimum radius
        self.r_inner = r_inner
        self.r_outer = r_outer

    def _hungarian(self, cost_matrix):
        """
            return ->   total cost of best assignment,
                        best assignment indexes
        """
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        total = cost_matrix[row_ind, col_ind].sum()
        indexes = zip(row_ind.tolist(), col_ind.tolist())
        return total, indexes

    def _lapjv(self, cost_matrix):
        """
            return ->   total cost of best assignment,
                        best assignment indexes
        """
        row_ind = list(range(cost_matrix.shape[0]))
        col_ind = lapjv(cost_matrix)[1]
        total = cost_matrix[row_ind, col_ind].sum()
        indexes = zip(row_ind, col_ind)
        return total, indexes

    def canny_edge_shape(self, img, max_samples=100, t1=100, t2=200):
        '''
            return -> list of sampled Points from edges
                      founded by canny edge detector.
        '''
        edges = cv2.Canny(img, t1, t2)
        x, y = np.where(edges != 0)
        if x.shape[0] > max_samples:
            idx = np.random.choice(x.shape[0], max_samples, replace=False)
            x, y = x[idx], y[idx]
        shape = []
        for i in range(x.shape[0]):
            shape.append([x[i], y[i]])
        return shape

    def _hist_cost(self, hi, hj):
        '''
            return -> cost of matching points with hi & hj
                      histograms respectively (chi-square test)
        '''
        cost = 0
        for k in range(self.nbins_theta * self.nbins_r):
            if (hi[k] + hj[k]):
                cost += ((hi[k] - hj[k])**2) / (hi[k] + hj[k])

        return cost * 0.5

    def _cost_matrix(self, P, Q):
        '''
            P       ->  Array of histograms associated with one shape
            Q       ->  Array of histograms associated with another shape
            return  ->  cost matrix of matching each
                        histogram in P with each one in Q
        '''
        p, _ = P.shape
        q, _ = Q.shape
        C = np.zeros((p, q))
        for i in range(p):
            for j in range(q):
                # Divide every histogram by number of points to
                # improve accuracy if they have different number of
                # points
                C[i, j] = self._hist_cost(P[i]/p, Q[j]/q)

        return C

    def cost(self, P, Q):
        '''
            P       ->  Array of histograms associated with one shape
            Q       ->  Array of histograms associated with another shape
            return  ->  cost of matching shape with P histograms
                        to shape with Q histograms
        '''
        C = self._cost_matrix(P,Q)
        cost, _= self._hungarian(C)

        return cost

    def compute(self, points):
        """
            return -> Array with the descriptors of every point
        """
        t_points = len(points)
        # getting euclidian distance
        r_array = cdist(points, points)

        # normalizing
        r_array_n = r_array / r_array.mean()
        # create log space
        r_bin_edges = np.logspace(np.log10(self.r_inner), np.log10(self.r_outer), self.nbins_r)
        r_array_q = np.zeros((t_points, t_points), dtype=int)
        # summing occurences in different log space intervals
        for m in range(self.nbins_r):
            r_array_q += (r_array_n < r_bin_edges[m])

        fz = r_array_q > 0

        # getting angles in radians
        theta_array = cdist(points, points, lambda u, v: math.atan2((v[1] - u[1]), (v[0] - u[0])))
        # removing all very small values because of float operation
        theta_array[np.abs(theta_array) < 1e-7] = 0

        # 2Pi shifted because we need angels in [0,2Pi]
        theta_array_2 = theta_array + 2 * math.pi * (theta_array < 0)
        # Simple Quantization
        theta_array_q = (1 + np.floor(theta_array_2 / (2 * math.pi / self.nbins_theta))).astype(int)

        # building point descriptor based on angle and distance
        nbins = self.nbins_theta * self.nbins_r
        descriptor = np.zeros((t_points, nbins))
        for i in range(t_points):
            sn = np.zeros((self.nbins_r, self.nbins_theta))
            for j in range(t_points):
                if (fz[i, j]):
                    sn[r_array_q[i, j] - 1, theta_array_q[i, j] - 1] += 1
            descriptor[i] = sn.flatten()

        return descriptor

