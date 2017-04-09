from sympy import *
import numpy as np
import math
import random

# dette python-scriptet er en del av numerikk-prosjektet, den delen som omhandler
# newtons multivariate-metoden. Vi estimerer koordinater utifra reisetiden til
# signalet (med en kunstig pålagt feil) og sammenlikner med de faktiske koordinatene

# konstanter

c = 299792.458
err = 10e-8
actual_pos = [0,0,6370,0]
earthrad = 6370

# en klasse som representerer en satellitt med kartetiske koordinater og tid/distanse t unna målepunktet

class Satellite:
    def __init__(self, x, y, z, t):
        self.x = x
        self.y = y
        self.z = z
        self.t = t

# implementasjon av newtons multivariate

def newtons_satellites(satellites, x0):

    # først definerer vi x,y,z,d som symbolske variabler
    # fordi vi gjør dette kan vi partiellderivere
    # med hensyn på disse variablene senere

    syms = (Symbol('x'), Symbol('y'), Symbol('z'), Symbol('d'))

    # lag en ligning for hver satellitt
    
    symfuns = list(map(lambda s : (syms[0] - s.x)**2 +\
                   (syms[1] - s.y)**2 +\
                   (syms[2] - s.z)**2 -\
                   (c * (s.t - syms[3]))**2, satellites))
    
    def F(vec):
        return list(map(lambda f : lambdify(syms, f)(vec[0], vec[1], vec[2], vec[3]), symfuns))

    # her konstruerer vi jacobi-matrisen ved å partiellderivere
    # hver av funksjonene fra symfuns med hensyn på hver av
    # de symbolse variablene, så vi får en kvadratisk matrise av funksjoner
    
    jacobi = list(map(lambda f : list(map(lambda s : lambdify(s, f.diff(s), 'numpy'), syms)), symfuns))

    def DF(vec): 
        return list(map(lambda f_vec : list(map(lambda i : f_vec[i](vec[i]), range(0, 4))),jacobi))

    df = []
    for i in range(0, 10): # bruk newtons metode med 10 iterasjoner
        df = DF(x0)
        x0 -= np.dot(np.linalg.inv(df), F(x0))
    return [x0, df] # returner svaret (som nå ligger i x0), og en numerisk jakobi-matrise (for debug)

# en wrapper til newton som tar inn polare koordinater og kunstig feil
# og konverterer til kartetiske koordinater

def newtons_satellites_polar(sat_data, x0, rho): # sat_data = [[phi, theta, err] {for hver satellitt}]
    return newtons_satellites(list(map(lambda data :
                                Satellite(rho*cos(data[0])*cos(data[1]),
                                          rho*cos(data[0])*sin(data[1]),
                                          rho*sin(data[0]),
                                          x0[3] + sqrt((rho*cos(data[0])*cos(data[1]))**2 +
                                                   (rho*cos(data[0])*sin(data[1]))**2 +
                                                       (rho*sin(data[0]) - x0[2])**2)/c + data[2]),
                                       sat_data)), x0)

def quadratic_formula(sats):
    s = sats[0] # første satellitt, bruker den mye, så greit å forkorte litt

    u = [[],[],[],[],[]] # [ux, uy, uz, ud, w]

    for i in range(0, 3):
        for j in range(0, 4):
            u[j].append(sats[3 - i][j] - s[j])
        u[4].append(c**2 * (s[3]**2 - sats[3 - i][3]**2) +
                            s[0]**2 - sats[3 - i][0]**2 +
                            s[1]**2 - sats[3 - i][1]**2 +
                            s[2]**2 - sats[3 - i][2]**2) 

    u[3] = list(map(lambda t : -2 * c**2 * t, u[3]))
    
    ax = np.linalg.det(np.matrix.transpose(np.array([u[1],u[2],u[0]])))
    bx = np.linalg.det(np.matrix.transpose(np.array([u[1],u[2],u[3]])))
    cx = np.linalg.det(np.matrix.transpose(np.array([u[1],u[2],u[4]])))

    ay = np.linalg.det(np.matrix.transpose(np.array([u[0],u[2],u[1]])))
    by = np.linalg.det(np.matrix.transpose(np.array([u[0],u[2],u[3]])))
    cy = np.linalg.det(np.matrix.transpose(np.array([u[0],u[2],u[4]])))

    az = np.linalg.det(np.matrix.transpose(np.array([u[0],u[1],u[2]])))
    bz = np.linalg.det(np.matrix.transpose(np.array([u[0],u[1],u[3]])))
    cz = np.linalg.det(np.matrix.transpose(np.array([u[0],u[1],u[4]])))

    abc_a = (bx/ax)**2 + (by/ay)**2 + (bz/az)**2 - c**2
    abc_b = 2 * (bx/ax) * (cx/ax + s[0]) + 2 * (by/ay) * (cy/ay+s[1]) + 2 * (bz/az+s[2]) + 2 * c**2 * s[3]
    abc_c = (cx/ax + s[0])**2 + (cy/ay+s[1])**2 + (cz/az+s[2])**2 - c**2 * s[3]**2

    if abc_b > 0:
        d1 = -((abc_b + sqrt(abc_b**2 - 4 * abc_a * abc_c)) / (2 * abc_a))
        d2 = -((2 * abc_c) / (abc_b + sqrt(abc_b**2 - 4 * abc_a * abc_c)))
    else:
        d1 = (-abc_b + sqrt(abc_b**2 - 4 * abc_a * abc_c)) / (2 * abc_a)
        d2 = (2 * abc_c) / (-abc_b + sqrt(abc_b**2 - 4 * abc_a * abc_c))

    return [[-(cx / ax + (bx / ax) * d1), -(cy / ay + (by / ay) * d1), -(cz / az + (bz / az) * d1), d1],
            [-(cx / ax + (bx / ax) * d2), -(cy / ay + (by / ay) * d2), -(cz / az + (bz / az) * d2), d2]]

# test 1: referansesatellitter fra oppgaven
# vi tester disse koordinatene og forsinkelsene for å se om det stemmer med fasit

satellites = [Satellite(15600, 7540, 20140, 0.07074),
              Satellite(18760, 2750, 18610, 0.07220),
              Satellite(17610, 14630, 13480, 0.07690),
              Satellite(19170, 610, 18390, 0.07242)]

init_vec = [5,5,earthrad,0]

print('\nsolution : ', newtons_satellites(satellites, init_vec)[0], '\n\n')
#print('quadratic formula', quadratic_formula(list(map(lambda s : [s.x,s.y,s.z,s.t],satellites))),'\n')

# test 2: feilmåling
# tester med vilkårlige satellittposisjoner
# vi har satt en feilmargin på 1e-8
# alle 81 kombinasjoner av feil sjekkes, så tar vi høyeste EMFaktor
# eksempel på kombinasjon: +1e-8 på t1, ingen endring på t2 og -1e-8 på t3 og t4

def find_emfs(data):
    emfs = [] # samle alle EMF i en liste

    # iterer gjennom alle kombinasjoner (treg)
    # (alle kombinasjoner av følgende elementer [-1, 0, 1])

    for e1 in range(-1, 2):
        for e2 in range(-1, 2):
            for e3 in range(-1, 2):
                for e4 in range(-1, 2):
                    data[0][2] = e1 * err
                    data[1][2] = e2 * err
                    data[2][2] = e3 * err
                    data[3][2] = e4 * err

                    # estimer posisjon med gitte feil
                    
                    approx = newtons_satellites_polar(data, init_vec, 26570)

                    # regn ut delta-verdier
                
                    for i in range(0, len(approx[0])):
                        approx[0][i] -= actual_pos[i]
                    
                    f_err = max(list(map(lambda val : abs(val),approx[0])))

                    delta_err = [e1 * err, e2 * err, e3 * err, e4 * err]

                    b_err = max(list(map(lambda val : abs(val), delta_err))) # (denne blir alltids err (untatt 1 gang))
                
                    if(b_err > 0): # (ett tilfelle hvor err er 0)
                        emfs.append(f_err/(c * b_err)) # emf = foroverfeil/(c*bakoverfeil)

    # finn kondisjonstall, altså største EMF
    for i in range(0, len(data)):
        print('satellitt #', (i + 1), ' phi = ', data[i][0], ' theta = ', data[i][1])
    return max(emfs)

# tester først med helt tilfeldige satellitter

random_data = [] # [phi, theta, error] for hver satellitt
close_data = [[0.211, 1.123, 0],[0.233, 1.213, 0],[0.192, 1.201, 0],[0.245, 1.101, 0]]
scattered_data = [[0.41,1.1,0],[0.53,2.2,0],[0.35,3.3,0],[0.42,4.4,0]]
#lined_data = []

for i in range(0, 4): # generer tilfeldige phi og theta for hver satellitt
    random_data.append([np.random.uniform(0, math.pi / 2), np.random.uniform(0, 2 * math.pi), 0])

print('\nkondisjonstall - tilfeldige satellitter :', find_emfs(random_data),'\n')
print('\nkondisjonstall - nære satellitter :', find_emfs(close_data),'\n')
print('\nkondisjonstall - spredte satellitter :', find_emfs(scattered_data),'\n')


# - - - Appendix Lambda - - - #

#     map(f, l)
# map tar to parametre, en funksjon f og en liste l
# deretter tar den hvert element i l og bruker de som
# input til funksjonen f, returverdien legges i en
# ny liste som returneres av map-funksjonen
# output-lista vil ha samme størrelse som l

#     lambda parametre : funksjonskropp
# lambda er en funksjon som returnerer anonyme funksjoner
# eksempel:
#     lambda x, y : 2 * x - y
# i eksempelet så har vi nå en anonym funksjon som tar
# to argumenter og returnerer en utregning av dem
# matte-ekvivalent: f(x, y) = 2x - y
# eneste forskjellen er at du ikke kan kalle på den
# ved å bruke f
# hvor enn du kan bruke en funksjon kan du heller bruke
# lambda hvis det ikke er noe poeng i å ha en referanse

# samlet eksempel
# la oss doble alle elementene i en liste
#
# input  >>> liste = [1, 2, 3]
# input  >>> list(map(lambda x : 2 * x, liste))
# output >>> [2, 4, 6]
#
# dette kan også gjøres med en anonym liste
# input  >>> list(map(lambda x : 2 * x, [1, 2, 3]))
# output >>> [2, 4, 6]
