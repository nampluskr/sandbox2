from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from time import clock
import cypct as cy
import matplotlib.patches as patches
from scipy.interpolate import interp1d
from numpy import cos, sin

########## Global constants for electrons:
E_CHARGE = 1.602E-19 # [C]
E_MASS   = 9.109E-31 # [Kg]

########## Electric Field [V/m]
class ElectricField:
    def __init__(self, voltage, sheath, target):
        self.vol = voltage
        self.sh = sheath
        self.tm = target
    def Ex(self,x,y,z): return np.zeros(z.size)
    def Ey(self,x,y,z): return np.zeros(z.size)
    def Ez(self,x,y,z):
        return (-np.sign(z-self.sh-self.tm)+1) \
                *self.vol*(self.sh+self.tm-z)/self.sh**2

########## Magnetic field [T] (Analytic expression)
class MagneticField:
    def __init__(self, *magnets):
        self.mags = magnets
    def Bx(self,x,y,z):
        sum_bx = 0.0
        for m in self.mags:
            sum_bx += m.Bx(x,y,z)
        return sum_bx
    def By(self,x,y,z):
        sum_by = 0.0
        for m in self.mags:
            sum_by += m.By(x,y,z)
        return sum_by
    def Bz(self,x,y,z):
        sum_bz = 0.0
        for m in self.mags:
            sum_bz += m.Bz(x,y,z)
        return sum_bz

class Magnet:
    def __init__(self,x0,y0,z0,length,width,height,scale):
        self.a, self.b, self.c  = length, width, height
        self.x0,self.y0,self.z0 = x0, y0, z0
        self.scale = scale
    def Bx(self,x,y,z):
        x, y, z = x-self.x0, y-self.y0, z-self.z0
        bx = f1(self.a-x,y,z,self.c) + f1(self.a-x,self.b-y,z,self.c) \
           - f1(x,y,z,self.c)        - f1(x,self.b-y,z,self.c) \
           - f1(self.a-x,y,z,0)      - f1(self.a-x,self.b-y,z,0) \
           + f1(x,y,z,0)             + f1(x,self.b-y,z,0)
        return -self.scale*bx*0.5
    def By(self,x,y,z):
        x, y, z = x-self.x0, y-self.y0, z-self.z0
        by = f1(self.b-y,x,z,self.c) + f1(self.b-y,self.a-x,z,self.c) \
           - f1(y,x,z,self.c)        - f1(y,self.a-x,z,self.c) \
           - f1(self.b-y,x,z,0)      - f1(self.b-y,self.a-x,z,0) \
           + f1(y,x,z,0)             + f1(y,self.a-x,z,0)
        return -self.scale*by*0.5
    def Bz(self,x,y,z):
        x, y, z = x-self.x0, y-self.y0, z-self.z0
        bz = f2(y,self.a-x,z,self.c) + f2(self.b-y,self.a-x,z,self.c) \
           + f2(x,self.b-y,z,self.c) + f2(self.a-x,self.b-y,z,self.c) \
           + f2(self.b-y,x,z,self.c) + f2(y,x,z,self.c) \
           + f2(self.a-x,y,z,self.c) + f2(x,y,z,self.c) \
           - f2(y,self.a-x,z,0)      - f2(self.b-y,self.a-x,z,0) \
           - f2(x,self.b-y,z,0)      - f2(self.a-x,self.b-y,z,0) \
           - f2(self.b-y,x,z,0)      - f2(y,x,z,0) \
           - f2(self.a-x,y,z,0)      - f2(x,y,z,0)
        return -self.scale*bz

def norm(x,y,z):
    return np.sqrt(x**2+y**2+z**2)

def f1(x,y,z,h):
    return np.log((norm(x,y,z-h)-y)/(norm(x,y,z-h)+y))

def f2(x,y,z,h):
    return np.arctan((x/y)*(z-h)/norm(x,y,z-h))

def eval_k(u, E, B):
    global E_CHARGE, E_MASS
    vx = u[3]
    vy = u[4]
    vz = u[5]
    ax = -(E.Ex(u[0],u[1],u[2]) + u[4]*B.Bz(u[0],u[1],u[2]) \
         - u[5]*B.By(u[0],u[1],u[2]))*(E_CHARGE/E_MASS)
    ay = -(E.Ey(u[0],u[1],u[2]) + u[5]*B.Bx(u[0],u[1],u[2]) \
         - u[3]*B.Bz(u[0],u[1],u[2]))*(E_CHARGE/E_MASS)
    az = -(E.Ez(u[0],u[1],u[2]) + u[3]*B.By(u[0],u[1],u[2]) \
         - u[4]*B.Bx(u[0],u[1],u[2]))*(E_CHARGE/E_MASS)
    return np.array([vx,vy,vz,ax,ay,az])

def solve_rk4(t, u0, E, B):
    uu = np.array([u0]*t.size)
    for i in xrange(t.size-1):
        h = t[i+1]-t[i]
        k1 = eval_k(uu[i], E, B)
        k2 = eval_k(uu[i] + k1*h/2., E, B)
        k3 = eval_k(uu[i] + k2*h/2., E, B)
        k4 = eval_k(uu[i] + k3*h, E, B)
        uu[i+1] = uu[i] + (k1+2*(k2+k3)+k4)*h/6.
    return uu

########## Particle tracing solver using 4th order Runge-Kutta method
class ElectronTrace:
    def __init__(self,t,u0):
        self.t = t
        self.u0 = u0
    def solve(self, E, B):
        self.u = solve_rk4(self.t, self.u0, E, B)
        return self.u


########## Particle tracing solver using 4th order Runge-Kutta method











########## Collisions
class CrossSection():
    def __init__(self, filename, torr, kelvin):
        data = np.genfromtxt(filename, delimiter=',', skip_header=3)
        self.sig_el = interp1d(data[:,0], data[:,1])
        self.sig_ex = interp1d(data[:,0], data[:,2])
        self.sig_iz = interp1d(data[:,0], data[:,3])
        self.sig_tt = interp1d(data[:,0], data[:,4])
        self.ng = torr*133.32*6.022E23/8.314/kelvin # [#/m^3]

    def prob_el(self, ke, distance):
        return 1-np.exp(-self.ng*self.sig_el(ke)*1.0E-20*distance)
    def prob_ex(self, ke, distance):
        return 1-np.exp(-self.ng*self.sig_ex(ke)*1.0E-20*distance)
    def prob_iz(self, ke, distance):
        return 1-np.exp(-self.ng*self.sig_iz(ke)*1.0E-20*distance)
    def prob_tt(self, ke, distance):
        return 1-np.exp(-self.ng*self.sig_tt(ke)*1.0E-20*distance)

    def collision(self, ke):
        sig_el = self.sig_el(ke)
        sig_ex = self.sig_ex(ke)
        sig_tt = self.sig_tt(ke)
        r = np.random.rand()
        
        if r<sig_el/sig_tt and ke>1:
            return 0 # Elastic
        if r<(sig_el+sig_ex)/sig_tt and ke>12:
            return 1 # Excitation
        if ke>16:
            return 2 # Ionization        

def new_velocity(vx,vy,vz,collision=0):
    vmag = np.sqrt(vx**2 + vy**2 + vz**2)
    ke = 0.5*E_MASS*vmag**2/E_CHARGE
    
    theta = np.arccos(vz/vmag)
    phi   = np.arctan(vy/vx)
    r     = np.random.rand()
    chi   = np.arccos(1.0-2*r/(1+8*ke*(1-r)/27.21))
    psi   = np.pi*2*r    

    energyloss = [1,12,16]
    new_ke = k2 - energyloss[collision]
    new_vmag = np.sqrt(2*new_ke*E_CHARGE/E_MASS)
    
    new_vx =  cos(theta)*cos(phi)*sin(chi)*cos(psi) \
             -sin(phi)*sin(chi)*sin(psi) + sin(theta)*cos(phi)*cos(chi)
    new_vy =  cos(theta)*sin(phi)*sin(chi)*cos(psi) \
             -cos(phi)*sin(chi)*sin(psi) + sin(theta)*sin(phi)*cos(chi)
    new_vz = -sin(theta)*sin(chi)*cos(psi) + cos(theta)**cos(chi)

    return np.array([new_vx,new_vy,new_vz])*new_vmag

if __name__ == "__main__":


    if 0: # Fig. 5.2
        ar = CrossSection("ArCrossSections.csv", torr=0.005, kelvin=293)
        ke = np.linspace(0,200,200)
        
        fig = plt.figure(figsize=(8,5))
        ax = fig.add_subplot(111)
        ax.plot(ke, ar.sig_el(ke)*1.0E-20, lw=2, label="Elastic")
        ax.plot(ke, ar.sig_ex(ke)*1.0E-20, lw=2, label="Excitation")
        ax.plot(ke, ar.sig_iz(ke)*1.0E-20, lw=2, label="Ionization")
        
        ax.set_title("(Fig. 5.2) Electron-Argon Cross Sections", weight="bold", size=15)
        ax.set_xlabel("Energy [eV]", weight="bold", size=11)
        ax.set_ylabel("Cross Section [m^2]", weight="bold", size=11)
        ax.grid(True)
        ax.legend(loc="best")
        
    if 0: # Fig. 5.3
        ar = CrossSection("ArCrossSections.csv", torr=0.005, kelvin=293)
        x = np.linspace(0,1,200)
        
        fig = plt.figure(figsize=(8,5))
        ax = fig.add_subplot(111)
        ax.plot(x, ar.prob_el(100,x), lw=2, label="Elastic")
        ax.plot(x, ar.prob_ex(100,x), lw=2, label="Excitation")
        ax.plot(x, ar.prob_iz(100,x), lw=2, label="Ionization")
        
        ax.set_title("(Fig. 5.3) Electron-Argon Collision Probability", weight="bold", size=15)
        ax.set_xlabel("Energy [eV]", weight="bold", size=11)
        ax.set_ylabel("Probability", weight="bold", size=11)
        ax.grid(True)
        ax.legend(loc="best")
        
    if 0: # Fig. 5.4
        ar = CrossSection("ArCrossSections.csv", torr=0.005, kelvin=293)
        x = np.linspace(0,1,200)
        
        fig = plt.figure(figsize=(8,5))
        ax = fig.add_subplot(111)
        ax.plot(x, ar.prob_iz(100,x), lw=2, label="100eV")
        ax.plot(x, ar.prob_iz(200,x), lw=2, label="200eV")
        ax.plot(x, ar.prob_iz(400,x), lw=2, label="400eV")
        
        ax.set_title("(Fig. 5.4) Electron-Argon Collision Probability", weight="bold", size=15)
        ax.set_xlabel("Energy [eV]", weight="bold", size=11)
        ax.set_ylabel("Probability", weight="bold", size=11)
        ax.grid(True)
        ax.legend(loc="best")
        
    if 0: # Fig. 5.5
        ar1 = CrossSection("ArCrossSections.csv", torr=0.002, kelvin=293)
        ar2 = CrossSection("ArCrossSections.csv", torr=0.005, kelvin=293)
        ar3 = CrossSection("ArCrossSections.csv", torr=0.010, kelvin=293)
        x = np.linspace(0,1,200)
        
        fig = plt.figure(figsize=(8,5))
        ax = fig.add_subplot(111)
        ax.plot(x, ar1.prob_iz(100,x), lw=2, label=" 2mtorr")
        ax.plot(x, ar2.prob_iz(100,x), lw=2, label=" 5mtorr")
        ax.plot(x, ar3.prob_iz(100,x), lw=2, label="10mtorr")
        
        ax.set_title("(Fig. 5.5) Electron-Argon Collision Probability", weight="bold", size=15)
        ax.set_xlabel("Energy [eV]", weight="bold", size=11)
        ax.set_ylabel("Probability", weight="bold", size=11)
        ax.grid(True)
        ax.legend(loc="best")
        
    if 0: # Fig. 5.7
        r = np.random.rand(100000)

        fig = plt.figure(figsize=(8,5))

        ke = 50
        angle = np.arccos(1.0-2*r/(1+8*ke*(1-r)/27.21))*180/np.pi
        hist, bins = np.histogram(angle, bins=200)
        width = 1.0*(bins[1]-bins[0])
        center = (bins[:-1]+bins[1:])/2.0
        ax = fig.add_subplot(111)
        ax.bar(center, hist, align="center", width=width, color='b', label="50eV")
       
        ke = 500
        angle = np.arccos(1.0-2*r/(1+8*ke*(1-r)/27.21))*180/np.pi
        hist, bins = np.histogram(angle, bins=200)
        width = 1.0*(bins[1]-bins[0])
        center = (bins[:-1]+bins[1:])/2.0
        ax = fig.add_subplot(111)
        ax.bar(center, hist, align="center", width=width, color='r', label="500eV")

        ax.set_title("(Fig. 5.7) Distribution of Axial Deflection", weight="bold", size=15)
        ax.set_xlabel("Angle [degrees]", weight="bold", size=11)
        ax.set_ylabel("Number", weight="bold", size=11)
        ax.grid(True)
        ax.legend(loc="best")

    if 1: # Determin Collision Type
        ar = CrossSection("ArCrossSections.csv", torr=0.005, kelvin=293)
        num = 200
        ke = np.linspace(0,200,num)
        coll_type = np.empty(num)
        for i in range(num):
            coll_type[i] = ar.collision(ke[i])
        
        fig = plt.figure(figsize=(8,5))
        ax = fig.add_subplot(111)
        ax.plot(ke, coll_type, 'o', lw=2, label="Elastic")
        ax.set_ylim(-1,3)
        
