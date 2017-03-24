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

def F(p_x,p_y,p_z,p_d):
    return  list(map(lambda f : lambdify((x,y,z,d), f)(p_x,p_y,p_z,p_d), symfuns))

DF_list = [];

for f in symfuns:
    DF_list.append([lambdify(x, f.diff(x), 'numpy'),
                    lambdify(y, f.diff(y), 'numpy'),
                    lambdify(z, f.diff(z), 'numpy'),
                    lambdify(d, f.diff(d), 'numpy')])

def DF(vec):
    out = []
    for f_vec in DF_list:
        tmp = []
        for i in range(0, 4):
            tmp.append(f_vec[i](vec[i]))    
        out.append(tmp)
    return out

x0 = [0,0,6370,0] # initial vector

for i in range(1, 11):
    x0 = x0 - np.dot(np.linalg.inv(DF(x0)), F(x0[0], x0[1], x0[2], x0[3]))
    print('iter #', i, ': ', x0)
