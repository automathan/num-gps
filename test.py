from sympy import *
import numpy as np

class Satellite:
    def __init__(self, x, y, z, t):
        self.x = x
        self.y = y
        self.z = z
        self.t = t

c = 299792.458

satellites = [Satellite(15600, 7540, 20140, 0.07074),
              Satellite(18760, 2750, 18610, 0.07220),
              Satellite(17610, 14630, 13480, 0.07690),
              Satellite(19170, 610, 18390, 0.07242)]

x = Symbol('x')
y = Symbol('y')
z = Symbol('z')
d = Symbol('d')

symfuns = []

for s in satellites:
    symfuns.append( (x - s.x)**2 +\
                    (y - s.y)**2 +\
                    (z - s.z)**2 -\
                    (c * (s.t - d))**2)
    
def F(p_x,p_y,p_z,p_d):
    return [lambdify((x,y,z,d), symfuns[0])(p_x,p_y,p_z,p_d),
            lambdify((x,y,z,d), symfuns[1])(p_x,p_y,p_z,p_d),
            lambdify((x,y,z,d), symfuns[2])(p_x,p_y,p_z,p_d),
            lambdify((x,y,z,d), symfuns[3])(p_x,p_y,p_z,p_d)]

DF_list = [];

for f in symfuns:
    DF_list.append([lambdify(x, f.diff(x), 'numpy'),
               lambdify(y, f.diff(y), 'numpy'),
               lambdify(z, f.diff(z), 'numpy'),
               lambdify(d, f.diff(d), 'numpy')])

def DF(p_x,p_y,p_z,p_d):
    out = []
    for f_vec in DF_list:
        out.append([f_vec[0](p_x),
                    f_vec[1](p_y),
                    f_vec[2](p_z),
                    f_vec[3](p_d)])
    return out

x0 = [0,0,6370,0]

for i in range(0, 5):
    x0 = x0 - np.dot(np.linalg.inv(DF(x0[0],x0[1],x0[2],x0[3])), F(x0[0],x0[1],x0[2],x0[3]))
    print('iter #', i, ': ', x0)
    
#ff = lambdify((x, y, z, d), funs[0])
#print(ff(1,2,3,4))
#f1prime = funs[0].diff(x)
#print(f1prime)
#f = lambdify(x, f1prime, 'numpy')
#print(f(5))


