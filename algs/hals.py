# Implemented based on the following paper:
# Bo Li, Guoxu Zhou, and Andrzej Cichocki. Two efficient algorithms for approximately orthogonal non-negative matrix factorization. IEEE Signal Processing Letters, 22(7):843–846, 2014.

import numpy as np
from .utils import divide, normalize

class HALS:
    name = "HALS"
    def func (X: np.ndarray, k: int, tol: float = 1e-4, max_iter: int = 1000):
        m, n = np.shape(X)
        lbd = 1
        A = (np.random.rand(m, k))
        B = (np.random.rand(n, k))
        B = normalize (B, 0)
        A_diff, B_diff = 1, 1
        period = 10
        count = 0
        iter=0
        while A_diff >= tol or B_diff >= tol:
            if count == 0:
                B_prev = np.copy(B)
                A_prev = np.copy(A)
            for __ in range(5):
                for r in range(k):
                    br = B[:, r]
                    A[:, r] = np.maximum(A[:, r] + X @ br - A @ (B.T @ br), 0)


            for __ in range(5):
                for r in range(k):
                    ones = np.ones(k)
                    ones[r] = 0
                    ar = A[:, r]
                    B[:, r] = np.maximum(B[:, r] + divide(X.T @ ar - B @ (A.T @ ar) - lbd * B @ ones, ar.T @ ar), 0)
            B = normalize (B, 0)
            if count == 0:
                A_diff = np.linalg.norm(A - A_prev, ord = "fro") /np.linalg.norm(A_prev, ord="fro")
                B_diff = np.linalg.norm(B - B_prev, ord = "fro") /np.linalg.norm(B_prev, ord="fro")
                
            count += 1
            if count >= period:
                count = 0
            if iter>max_iter:
                break
            iter=iter+1
        return A, B
