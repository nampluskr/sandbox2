from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from time import clock
import cypct as cy
import matplotlib.patches as patches
from scipy.interpolate import interp1d

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

########## Collision

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
        sig_iz = self.sig_iz(ke)
        sig_tt = self.sig_tt(ke)
        r = np.random.rand()
        
        if r<sig_el/sig_tt and ke>1:
            return 1.0 # Elastic
        elif r<(sig_el+sig_ex)/sig_tt and ke>12:
            return 2.0 # Excitation
        elif ke>16:
            return 3.0 # Ionization   

if __name__ == "__main__":


##############################################################################
    if 1: # Fig. 6.4
        tstart = clock()
        E = cy.ElectricField(-300.0, 0.001, 0.0)
        scale = 1.4/4.0/np.pi
        mag1 = cy.Magnet(-0.08, -0.01, -0.02, 0.16, 0.02, 0.01, scale) # Center
        mag2 = cy.Magnet(-0.10, -0.02, -0.02, 0.01, 0.04, 0.01,-scale) # Bottom
        mag3 = cy.Magnet(-0.10,  0.02, -0.02, 0.20, 0.01, 0.01,-scale) # Right
        mag4 = cy.Magnet( 0.09, -0.02, -0.02, 0.01, 0.04, 0.01,-scale) # Top
        mag5 = cy.Magnet(-0.10, -0.03, -0.02, 0.20, 0.01, 0.01,-scale) # Left
        mag_array = cy.MagneticField(mag1, mag2, mag3, mag4, mag5)

        t = np.arange(0.0,200.0,0.002)*1.0E-9 # [ns]  Tracing time
        x0, y0, z0  = (0.0,0.005,0.0)   # [m]   Initial position
        vx0,vy0,vz0 = (0.,0.,0.)         # [m/s] Initial velocity
        u0 = np.array([x0,y0,z0,vx0,vy0,vz0])

        pct = cy.ElectronTrace(t,u0)
        u = pct.solve(E, mag_array)
        e_x, e_y, e_z = u[:,0], u[:,1], u[:,2]
        ev_x, ev_y, ev_z = u[:,3], u[:,4], u[:,5]

        tend = clock()
        print "Time >> %.4e [sec]" % (tend-tstart), t.size

        xlim = [0.0,0.105]
        ylim = [-0.035,0.035]

        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)

        ax.add_patch(patches.Rectangle((-0.08,-0.01),0.16,0.02,\
                facecolor="gray", alpha=0.3 ))
        ax.add_patch(patches.Rectangle((-0.10,-0.02),0.01,0.04,\
                facecolor="gray", alpha=0.3 ))
        ax.add_patch(patches.Rectangle((-0.10, 0.02),0.20,0.01,\
                facecolor="gray", alpha=0.3 ))
        ax.add_patch(patches.Rectangle(( 0.09,-0.02),0.01,0.04,\
                facecolor="gray", alpha=0.3 ))
        ax.add_patch(patches.Rectangle((-0.10,-0.03),0.20,0.01,\
                facecolor="gray", alpha=0.3 ))

        ax.plot(e_x, e_y, lw=1.5)

        ax.set_title("(Fig. 6.4) Electron Trajectory Starting at (0,0.005)", weight="bold", size=15)
        ax.set_xlabel("X [m]", weight="bold", size=11); ax.set_xlim(xlim)
        ax.set_ylabel("Y [m]", weight="bold", size=11); ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.grid(True)
