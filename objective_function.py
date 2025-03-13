import numpy as np

# 目的関数
def objective_function(X,Y):
    t1 = 20
    t2 = -20 * np.exp(-0.2 * np.sqrt(1.0 / 2 * (X**2 + Y**2 )))
    t3 = np.e
    t4 = -np.exp(1.0 / 2 * (np.cos(2 * np.pi * X)+np.cos(2 * np.pi * Y)))
    return t1 + t2 + t3 + t4

def sphere_objective_function(x, y, z):
    """ 
    3次元Sphere関数。
    (x, y, z) = (0, 0, 0) のとき f(x, y, z) = 0
    """
    return x**2 + y**2 + z**2