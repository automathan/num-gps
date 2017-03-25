from sympy import *
import numpy as np

class Satellite:
    def __init__(self, x, y, z, t):
        self.x = x
        self.y = y
        self.z = z
        self.t = t

c = 299792.458 # speed of light

satellites = [Satellite(15600, 7540, 20140, 0.07074),
              Satellite(18760, 2750, 18610, 0.07220),
              Satellite(17610, 14630, 13480, 0.07690),
              Satellite(19170, 610, 18390, 0.07242)]

x = Symbol('x')
y = Symbol('y')
z = Symbol('z')
d = Symbol('d')

symfuns = list(map(lambda s : (x - s.x)**2 +\
                              (y - s.y)**2 +\
                              (z - s.z)**2 -\
                              (c * (s.t - d))**2, satellites))

def F(vec):
    return list(map(lambda f : lambdify((x,y,z,d), f)(vec[0],vec[1],vec[2],vec[3]), symfuns))

jacobi = list(map(lambda f : list(map(lambda s : lambdify(s, f.diff(s), 'numpy'), [x,y,z,d])), symfuns))

def DF(vec):
    return list(map(lambda f_vec : list(map(lambda i : f_vec[i](vec[i]), range(0, 4))),jacobi))

x0 = [0,0,6370,0] # initial vector

for i in range(0, 10):
    x0 = x0 - np.dot(np.linalg.inv(DF(x0)), F(x0))
    print('iter #', i, ': ', x0)
