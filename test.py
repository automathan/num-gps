from sympy import *
import numpy as np
import math


c = 299792.458

class Satellite:
    def __init__(self, x, y, z, t):
        self.x = x
        self.y = y
        self.z = z
        self.t = t
        
def newtons_satellites(satellites, x0):
    syms = (Symbol('x'), Symbol('y'), Symbol('z'), Symbol('d'))

    symfuns = list(map(lambda s : (syms[0] - s.x)**2 +\
                   (syms[1] - s.y)**2 +\
                   (syms[2] - s.z)**2 -\
                   (c * (s.t - syms[3]))**2, satellites))
    def F(vec):
        return list(map(lambda f : lambdify(syms, f)(vec[0], vec[1], vec[2], vec[3]), symfuns))

    jacobi = list(map(lambda f : list(map(lambda s : lambdify(s, f.diff(s), 'numpy'), syms)), symfuns))

    def DF(vec):
        return list(map(lambda f_vec : list(map(lambda i : f_vec[i](vec[i]), range(0, 4))),jacobi))
    
    for i in range(0, 10):
        x0 -= np.dot(np.linalg.inv(DF(x0)), F(x0))
        print('iter #', i, ': ', x0)
    return x0

def newtons_satellites_spherical(coords, x0, rho):
    return newtons_satellites(list(map(lambda coord :
                                Satellite(rho*cos(coord[0])*cos(coord[1]),
                                          rho*cos(coord[0])*sin(coord[1]),
                                          rho*sin(coord[0]),
                                          x0[3] + sqrt((rho*cos(coord[0])*cos(coord[1]))**2 +
                                                   (rho*cos(coord[0])*sin(coord[1]))**2 +
                                                   (rho*sin(coord[0]) - x0[2])**2)/c),
                                       coords)), x0)

satellites = [Satellite(15600, 7540, 20140, 0.07074),
              Satellite(18760, 2750, 18610, 0.07220),
              Satellite(17610, 14630, 13480, 0.07690),
              Satellite(19170, 610, 18390, 0.07242)]

init_vec = [0,0,6370,0]

print('\nsolution : ', newtons_satellites(satellites, init_vec), '\n\n')

coords = []

for i in range(0, 4):
    coords.append([np.random.uniform(0, math.pi / 2), np.random.uniform(0, 2 * math.pi)])

init_vec = [0,0,6370,0.0001]

print('\nsolution : ', newtons_satellites_spherical(coords, init_vec, 26570))

