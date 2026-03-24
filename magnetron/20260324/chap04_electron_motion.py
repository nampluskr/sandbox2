from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from time import clock
import cypct as cy
import matplotlib.patches as patches

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
    for i in range(t.size-1):
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

if __name__ == "__main__":

##############################################################################
    if 0: # Electric Field: cython vs. python
        E1 = cy.ElectricField(-300.0, 0.001, 0.0)
        E2 = ElectricField(-300.0, 0.001, 0.0)

        x = np.linspace(0.0, 0.002,100)
        y1 = np.ones(len(x))
        y2 = np.ones(len(x))

        for i, xi in enumerate(x):
            y1[i] = E1.Ez(0.0,0.0,xi)
            y2[i] = E2.Ez(0.0,0.0,xi)

        fig, ax = plt.subplots()
        ax.plot(x,y1,'bo')
        ax.plot(x,y2,'k')

        plt.show()

##############################################################################
    if 0: # Fig. 4.2 Field lines from two magnets
        scale = 1.4/4.0/np.pi
        mag1 = Magnet(-0.2, -0.02, -0.02, 0.4, 0.01, 0.01,  scale) # Left
        mag2 = Magnet(-0.2,  0.01, -0.02, 0.4, 0.01, 0.01, -scale) # Right
        mag_array = MagneticField(mag1, mag2)

        ylim = [-0.03,0.03]
        zlim = [-0.02,0.02]
        y = np.linspace(ylim[0], ylim[1], 500)
        z = np.linspace(zlim[0], zlim[1], 300)
        Y, Z = np.meshgrid(y,z)

        meshBy = mag_array.By(0.0, Y, Z)*10000
        meshBz = mag_array.Bz(0.0, Y, Z)*10000
        meshB  = np.sqrt(meshBy**2 + meshBz**2)

        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)

        ax.add_patch(patches.Rectangle((-0.02,-0.02),0.01,0.01,\
                facecolor="gray", alpha=0.3 ))
        ax.add_patch(patches.Rectangle((0.01,-0.02),0.01,0.01,\
                facecolor="gray", alpha=0.3 ))

        #ax.contourf(Y,Z,meshB, cmap = plt.cm.jet, \
        #    levels=np.linspace(100,5000,50))
        ax.contourf(Y,Z,meshBz, levels=np.linspace(-1,1,3), colors='k')
        ax.streamplot(Y,Z,meshBy,meshBz, \
            cmap = plt.cm.jet, color = np.clip(meshB,200,1500), \
            linewidth=1.5, density=1 )
        ax.axhline(y=0, c='k', lw=1.5, ls="--")

        ax.set_title("(Fig. 4.2) Field from Two Magnets", weight="bold", size=15)
        ax.set_xlabel("Y [m]", weight="bold", size=11); ax.set_xlim(ylim)
        ax.set_ylabel("Z [m]", weight="bold", size=11); ax.set_ylim(zlim)

        ax.set_aspect("equal")
        ax.grid(True)
        fig.tight_layout()
        plt.show()

##############################################################################
    if 0: # Fig. 4.3 Linear electric field in sheath
        E = ElectricField(-300.0, 0.001, 0.0)

        xlim = [0.0,0.002]
        ylim = [-600000,100000]
        x = np.linspace(xlim[0], xlim[1], 100)
        y = E.Ez(0.0, 0.0, x)

        fig = plt.figure(figsize=(8,5))
        ax = fig.add_subplot(111)

        ax.plot(x, y, lw=2)
        ax.axvline(x=0.001, c='k', lw=1.5, ls="--")

        ax.set_title("(Fig. 4.3) Linear Electric Field in Sheath", weight="bold", size=15)
        ax.set_xlabel("Z [m]", weight="bold", size=11); ax.set_xlim(xlim)
        ax.set_ylabel("Ez [V/m]", weight="bold", size=11); ax.set_ylim(ylim)

        ax.grid(True)
        fig.tight_layout()
        plt.show()

##############################################################################
    if 0: # Fig. 4.4 / Fig. 4.5 / Fig. 4.11
        tstart = clock()
        E = cy.ElectricField(-300.0, 0.001, 0.0)
        scale = 1.4/4.0/np.pi
        mag1 = cy.Magnet(-0.2, -0.02, -0.02, 0.4, 0.01, 0.01,  scale) # Left
        mag2 = cy.Magnet(-0.2,  0.01, -0.02, 0.4, 0.01, 0.01, -scale) # Right
        mag_array = cy.MagneticField(mag1, mag2)

        t = np.arange(0.0,100.0,0.01)*1.0E-9 # [ns]  Tracing time
        x0, y0, z0  = (0.0,-0.007,0.0)   # [m]   Initial position
        vx0,vy0,vz0 = (0.,0.,0.)         # [m/s] Initial velocity
        u0 = np.array([x0,y0,z0,vx0,vy0,vz0])

        pct = cy.ElectronTrace(t,u0)
        u = pct.solve(E, mag_array)
        e_x, e_y, e_z = u[:,0], u[:,1], u[:,2]
        ev_x, ev_y, ev_z = u[:,3], u[:,4], u[:,5]
        tend = clock()

        print "Time >> %.4e [sec]" % (tend-tstart), t.size

        xlim = [0.0,0.1]
        ylim = [-0.01,0.01]
        zlim = [0.0,0.0025]

        fig1 = plt.figure(figsize=(8,5))
        ax1 = fig1.add_subplot(211)
        ax1.plot(e_x, e_y, lw=1.5)

        ax1.set_title("(Fig. 4.4) Electron Trajectory in X-Y Plane", weight="bold", size=15)
        ax1.set_xlabel("X [m]", weight="bold", size=11); ax1.set_xlim(xlim)
        ax1.set_ylabel("Y [m]", weight="bold", size=11); ax1.set_ylim(ylim)
        ax1.set_aspect("equal")
        ax1.grid(True)

        y = np.linspace(ylim[0], ylim[1], 500)
        z = np.linspace(zlim[0], zlim[1], 300)
        Y, Z = np.meshgrid(y,z)

        scale = 1.4/4.0/np.pi
        m1 = Magnet(-0.2, -0.02, -0.02, 0.4, 0.01, 0.01,  scale) # Left
        m2 = Magnet(-0.2,  0.01, -0.02, 0.4, 0.01, 0.01, -scale) # Right
        m_array = MagneticField(m1, m2)

        meshBy = m_array.By(0.0, Y, Z)*10000
        meshBz = m_array.Bz(0.0, Y, Z)*10000
        meshB  = np.sqrt(meshBy**2 + meshBz**2)

        ax2 = fig1.add_subplot(212)
        ax2.plot(e_y,e_z,lw=1.5)
        ax2.streamplot(Y,Z,meshBy,meshBz, \
            cmap = plt.cm.jet, color = np.clip(meshB,200,1500), \
            linewidth=1.5, density=1)

        ax2.set_title("(Fig. 4.5) Side View Electron Trajectory", weight="bold", size=15)
        ax2.set_xlabel("Y [m]", weight="bold", size=11); ax2.set_xlim(ylim)
        ax2.set_ylabel("Z [m]", weight="bold", size=11); ax2.set_ylim(zlim)
        ax1.set_aspect("equal")
        ax2.grid(True)
        fig1.tight_layout()

        fig2 = plt.figure(figsize=(8,10))

        ax1 = fig2.add_subplot(311)
        ax1.plot(e_x, ev_x, lw=1.5)
        ax1.set_title("Phase Space: Vx vs. X", weight="bold", size=15)
        ax1.set_xlabel("X [m]", weight="bold", size=11); ax1.set_xlim([0,0.014])
        ax1.set_ylabel("Vx [m/s]", weight="bold", size=11)
        ax1.grid(True)

        ax2 = fig2.add_subplot(312)
        ax2.plot(e_y, ev_y, lw=1.5)
        ax2.set_title("Phase Space: Vy vs. Y", weight="bold", size=15)
        ax2.set_xlabel("Y [m]", weight="bold", size=11); ax2.set_xlim([-0.008,0.008])
        ax2.set_ylabel("Vy [m/s]", weight="bold", size=11)
        ax2.grid(True)

        ax3 = fig2.add_subplot(313)
        ax3.plot(e_z, ev_z, lw=1.5)
        ax3.set_title("Phase Space: Vz vs. Z", weight="bold", size=15)
        ax3.set_xlabel("Z [m]", weight="bold", size=11); ax3.set_xlim([0.0,0.0022])
        ax3.set_ylabel("Vz [m/s]", weight="bold", size=11)
        ax3.grid(True)

        fig2.tight_layout()
        plt.show()

##############################################################################
    if 0: # Fig. 4.19 / Fig. 4.20
        tstart = clock()
        E = cy.ElectricField(-300.0, 0.001, 0.0)
        scale = 1.4/4.0/np.pi
        mag1 = cy.Magnet(-0.2, -0.02, -0.02, 0.2, 0.01, 0.01, scale/2.0) # Left
        mag2 = cy.Magnet( 0.0, -0.02, -0.02, 0.2, 0.01, 0.01, scale) # Left
        mag3 = cy.Magnet(-0.2,  0.01, -0.02, 0.2, 0.01, 0.01,-scale/2.0) # Right
        mag4 = cy.Magnet( 0.0,  0.01, -0.02, 0.2, 0.01, 0.01,-scale) # Right
        mag_array = cy.MagneticField(mag1, mag2, mag3, mag4)

        t = np.arange(0.0,150.0,0.01)*1.0E-9 # [ns]  Tracing time
        x0, y0, z0  = (-0.1,-0.007,0.0)   # [m]   Initial position
        vx0,vy0,vz0 = (0.,0.,0.)         # [m/s] Initial velocity
        u0 = np.array([x0,y0,z0,vx0,vy0,vz0])

        pct = cy.ElectronTrace(t,u0)
        u = pct.solve(E, mag_array)
        e_x, e_y, e_z = u[:,0], u[:,1], u[:,2]
        ev_x, ev_y, ev_z = u[:,3], u[:,4], u[:,5]
        tend = clock()

        print "Time >> %.4e [sec]" % (tend-tstart), t.size

        xlim = [-0.1,0.05]
        ylim = [-0.01,0.01]
        zlim = [0.0,0.0035]

        fig1 = plt.figure(figsize=(8,5))
        ax1 = fig1.add_subplot(211)
        ax1.plot(e_x, e_y, lw=1.5)

        ax1.set_title("(Fig. 4.19) Electron Trajectory in X-Y Plane", weight="bold", size=15)
        ax1.set_xlabel("X [m]", weight="bold", size=11); ax1.set_xlim(xlim)
        ax1.set_ylabel("Y [m]", weight="bold", size=11); ax1.set_ylim(ylim)
        ax1.set_aspect("equal")
        ax1.grid(True)

        y = np.linspace(ylim[0], ylim[1], 500)
        z = np.linspace(zlim[0], zlim[1], 300)
        Y, Z = np.meshgrid(y,z)

        scale = 1.4/4.0/np.pi
        m1 = Magnet(-0.2, -0.02, -0.02, 0.4, 0.01, 0.01,  scale) # Left
        m2 = Magnet(-0.2,  0.01, -0.02, 0.4, 0.01, 0.01, -scale) # Right
        m_array = MagneticField(m1, m2)

        meshBy = m_array.By(0.0, Y, Z)*10000
        meshBz = m_array.Bz(0.0, Y, Z)*10000
        meshB  = np.sqrt(meshBy**2 + meshBz**2)

        ax2 = fig1.add_subplot(212)
        ax2.plot(e_y,e_z,lw=1.5)
        ax2.streamplot(Y,Z,meshBy,meshBz, \
            cmap = plt.cm.jet, color = np.clip(meshB,200,1500), \
            linewidth=1.5, density=1)

        ax2.set_title("(Fig. 4.20) Side View Electron Trajectory", weight="bold", size=15)
        ax2.set_xlabel("Y [m]", weight="bold", size=11); ax2.set_xlim(ylim)
        ax2.set_ylabel("Z [m]", weight="bold", size=11); ax2.set_ylim(zlim)
        ax1.set_aspect("equal")
        ax2.grid(True)
        fig1.tight_layout()

        plt.show()

##############################################################################
    if 0: # Fig. 4.24
        scale = 1.4/4.0/np.pi
        mag1 = Magnet(-0.175, -0.010, -0.02, 0.35, 0.02, 0.01, scale) # Center
        mag2 = Magnet(-0.200, -0.025, -0.02, 0.01, 0.05, 0.01,-scale) # Bottom
        mag3 = Magnet(-0.200,  0.025, -0.02, 0.40, 0.01, 0.01,-scale) # Right
        mag4 = Magnet( 0.190, -0.025, -0.02, 0.01, 0.05, 0.01,-scale) # Top
        mag5 = Magnet(-0.200, -0.035, -0.02, 0.40, 0.01, 0.01,-scale) # Left
        mag_array = MagneticField(mag1, mag2, mag3, mag4, mag5)

        ylim = [-0.04,0.04]
        zlim = [-0.02,0.03]
        y = np.linspace(ylim[0], ylim[1], 500)
        z = np.linspace(zlim[0], zlim[1], 300)
        Y, Z = np.meshgrid(y,z)

        meshBx = mag_array.Bx(0.0, Y, Z)*10000
        meshBy = mag_array.By(0.0, Y, Z)*10000
        meshBz = mag_array.Bz(0.0, Y, Z)*10000
        meshB  = np.sqrt(meshBy**2 + meshBz**2)

        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)

        ax.add_patch(patches.Rectangle((-0.035,-0.02),0.01,0.01,\
                facecolor="gray", alpha=0.3 ))
        ax.add_patch(patches.Rectangle((-0.010,-0.02),0.02,0.01,\
                facecolor="gray", alpha=0.3 ))
        ax.add_patch(patches.Rectangle((0.025,-0.02),0.01,0.01,\
                facecolor="gray", alpha=0.3 ))

        #ax.contourf(Y,Z,meshB, cmap = plt.cm.jet, \
        #    levels=np.linspace(50,8000,20))
        ax.contourf(Y,Z,meshBz, levels=np.linspace(-1,1,3), colors='k')
        ax.streamplot(Y,Z,meshBy,meshBz, \
            cmap = plt.cm.jet, color = np.clip(meshB,200,1500), \
            linewidth=1.5, density=1 )
        ax.axhline(y=0, c='k', lw=1.5, ls="--")

        ax.set_title("(Fig. 4.24) Magnetic Field @x=0cm", weight="bold", size=15)
        ax.set_xlabel("Y [m]", weight="bold", size=11); ax.set_xlim(ylim)
        ax.set_ylabel("Z [m]", weight="bold", size=11); ax.set_ylim(zlim)

        ax.set_aspect("equal")
        ax.grid(True)
        fig.tight_layout()
        plt.show()

##############################################################################
    if 0: # Fig. 4.25
        tstart = clock()
        E = cy.ElectricField(-300.0, 0.001, 0.0)
        scale = 1.4/4.0/np.pi
        mag1 = cy.Magnet(-0.175, -0.010, -0.02, 0.35, 0.02, 0.01, scale) # Center
        mag2 = cy.Magnet(-0.200, -0.025, -0.02, 0.01, 0.05, 0.01,-scale) # Bottom
        mag3 = cy.Magnet(-0.200,  0.025, -0.02, 0.40, 0.01, 0.01,-scale) # Right
        mag4 = cy.Magnet( 0.190, -0.025, -0.02, 0.01, 0.05, 0.01,-scale) # Top
        mag5 = cy.Magnet(-0.200, -0.035, -0.02, 0.40, 0.01, 0.01,-scale) # Left
        mag_array = cy.MagneticField(mag1, mag2, mag3, mag4, mag5)

        t = np.arange(0.0,260.0,0.01)*1.0E-9 # [ns]  Tracing time
        x0, y0, z0  = (-0.1,-0.007,0.0)   # [m]   Initial position
        vx0,vy0,vz0 = (0.,0.,0.)         # [m/s] Initial velocity
        u0 = np.array([x0,y0,z0,vx0,vy0,vz0])

        pct = cy.ElectronTrace(t,u0)
        u = pct.solve(E, mag_array)
        e_x, e_y, e_z = u[:,0], u[:,1], u[:,2]
        ev_x, ev_y, ev_z = u[:,3], u[:,4], u[:,5]

        tend = clock()
        print "Time >> %.4e [sec]" % (tend-tstart), t.size

        xlim = [-0.205,-0.1]
        ylim = [-0.04,0.04]

        fig1 = plt.figure(figsize=(8,6))
        ax1 = fig1.add_subplot(111)

        ax1.add_patch(patches.Rectangle((-0.175,-0.010),0.35,0.02,\
                facecolor="gray", alpha=0.3 ))
        ax1.add_patch(patches.Rectangle((-0.200,-0.035),0.40,0.01,\
                facecolor="gray", alpha=0.3 ))
        ax1.add_patch(patches.Rectangle((-0.200,0.025),0.40,0.01,\
                facecolor="gray", alpha=0.3 ))
        ax1.add_patch(patches.Rectangle((0.190,-0.025),0.01,0.05,\
                facecolor="gray", alpha=0.3 ))
        ax1.add_patch(patches.Rectangle((-0.200,-0.025),0.01,0.05,\
                facecolor="gray", alpha=0.3 ))

        ax1.plot(e_x, e_y, lw=1.5)

        ax1.set_title("(Fig. 4.25) Electron Trajectory Starting at (-0.1,-0.007)", weight="bold", size=15)
        ax1.set_xlabel("X [m]", weight="bold", size=11); ax1.set_xlim(xlim)
        ax1.set_ylabel("Y [m]", weight="bold", size=11); ax1.set_ylim(ylim)
        ax1.set_aspect("equal")
        ax1.grid(True)

##############################################################################
    if 0: # Fig. 4.26
        tstart = clock()
        E = cy.ElectricField(-300.0, 0.001, 0.0)
        scale = 1.4/4.0/np.pi
        mag1 = cy.Magnet(-0.175, -0.010, -0.02, 0.35, 0.02, 0.01, scale) # Center
        mag2 = cy.Magnet(-0.200, -0.025, -0.02, 0.01, 0.05, 0.01,-scale) # Bottom
        mag3 = cy.Magnet(-0.200,  0.025, -0.02, 0.40, 0.01, 0.01,-scale) # Right
        mag4 = cy.Magnet( 0.190, -0.025, -0.02, 0.01, 0.05, 0.01,-scale) # Top
        mag5 = cy.Magnet(-0.200, -0.035, -0.02, 0.40, 0.01, 0.01,-scale) # Left
        mag_array = cy.MagneticField(mag1, mag2, mag3, mag4, mag5)

        t = np.arange(0.0,260.0,0.01)*1.0E-9 # [ns]  Tracing time

        x0, y0, z0  = (-0.1,-0.007,0.0)   # [m]   Initial position
        vx0,vy0,vz0 = (0.,0.,0.)          # [m/s] Initial velocity
        u0 = np.array([x0,y0,z0,vx0,vy0,vz0])
        pct = cy.ElectronTrace(t,u0)
        u = pct.solve(E, mag_array)
        e_x, e_y, e_z = u[:,0], u[:,1], u[:,2]

        tend = clock()
        print "Time >> %.4e [sec]" % (tend-tstart), t.size

        xlim = [-0.205,-0.1]
        ylim = [-0.04,0.04]
        x = np.linspace(xlim[0], xlim[1], 200)
        y = np.linspace(ylim[0], ylim[1], 100)
        X, Y = np.meshgrid(x,y)

        m1 = Magnet(-0.175, -0.010, -0.02, 0.35, 0.02, 0.01, scale) # Center
        m2 = Magnet(-0.200, -0.025, -0.02, 0.01, 0.05, 0.01,-scale) # Bottom
        m3 = Magnet(-0.200,  0.025, -0.02, 0.40, 0.01, 0.01,-scale) # Right
        m4 = Magnet( 0.190, -0.025, -0.02, 0.01, 0.05, 0.01,-scale) # Top
        m5 = Magnet(-0.200, -0.035, -0.02, 0.40, 0.01, 0.01,-scale) # Left
        m_array = MagneticField(m1, m2, m3, m4, m5)

        meshBx = m_array.Bx(X, Y, 0.0)*10000
        meshBy = m_array.By(X, Y, 0.0)*10000
        meshBz = m_array.Bz(X, Y, 0.0)*10000
        meshB  = np.sqrt(meshBx**2 + meshBy**2)

        fig1 = plt.figure(figsize=(8,6))
        ax1 = fig1.add_subplot(111)

        ax1.contourf(X,Y,meshB, cmap = plt.cm.jet)
        ax1.contourf(X,Y,meshBz, levels=np.linspace(-10,10,3), colors='k')
        #ax1.streamplot(X,Y,meshBx,meshBy, \
        #    cmap = plt.cm.jet, color = np.clip(meshB,200,1500), \
        #    linewidth=1, density=2)

        ax1.add_patch(patches.Rectangle((-0.175,-0.010),0.35,0.02,\
                facecolor="gray", alpha=0.3 ))
        ax1.add_patch(patches.Rectangle((-0.200,-0.035),0.40,0.01,\
                facecolor="gray", alpha=0.3 ))
        ax1.add_patch(patches.Rectangle((-0.200,0.025),0.40,0.01,\
                facecolor="gray", alpha=0.3 ))
        ax1.add_patch(patches.Rectangle((0.190,-0.025),0.01,0.05,\
                facecolor="gray", alpha=0.3 ))
        ax1.add_patch(patches.Rectangle((-0.200,-0.025),0.01,0.05,\
                facecolor="gray", alpha=0.3 ))

        ax1.plot(e_x, e_y, lw=1.5)

        ax1.set_title("(Fig. 4.26) Electron Trajectory Starting at (-0.1,-0.007)", weight="bold", size=15)
        ax1.set_xlabel("X [m]", weight="bold", size=11); ax1.set_xlim(xlim)
        ax1.set_ylabel("Y [m]", weight="bold", size=11); ax1.set_ylim(ylim)
        ax1.set_aspect("equal")
        ax1.grid(True)

##############################################################################
    if 0: # Fig. 4.25 Starting at (-0.1,-0.01)
        tstart = clock()
        E = cy.ElectricField(-300.0, 0.001, 0.0)
        scale = 1.4/4.0/np.pi
        mag1 = cy.Magnet(-0.175, -0.010, -0.02, 0.35, 0.02, 0.01, scale) # Center
        mag2 = cy.Magnet(-0.200, -0.025, -0.02, 0.01, 0.05, 0.01,-scale) # Bottom
        mag3 = cy.Magnet(-0.200,  0.025, -0.02, 0.40, 0.01, 0.01,-scale) # Right
        mag4 = cy.Magnet( 0.190, -0.025, -0.02, 0.01, 0.05, 0.01,-scale) # Top
        mag5 = cy.Magnet(-0.200, -0.035, -0.02, 0.40, 0.01, 0.01,-scale) # Left
        mag_array = cy.MagneticField(mag1, mag2, mag3, mag4, mag5)

        t = np.arange(0.0,260.0,0.01)*1.0E-9 # [ns]  Tracing time
        x0, y0, z0  = (-0.1,-0.01,0.0)   # [m]   Initial position
        vx0,vy0,vz0 = (0.,0.,0.)         # [m/s] Initial velocity
        u0 = np.array([x0,y0,z0,vx0,vy0,vz0])

        pct = cy.ElectronTrace(t,u0)
        u = pct.solve(E, mag_array)
        e_x, e_y, e_z = u[:,0], u[:,1], u[:,2]
        ev_x, ev_y, ev_z = u[:,3], u[:,4], u[:,5]

        tend = clock()
        print "Time >> %.4e [sec]" % (tend-tstart), t.size

        xlim = [-0.205,0.0]
        ylim = [-0.04,0.04]

        fig1 = plt.figure(figsize=(8,6))
        ax1 = fig1.add_subplot(111)

        ax1.add_patch(patches.Rectangle((-0.175,-0.010),0.35,0.02,\
                facecolor="gray", alpha=0.3 ))
        ax1.add_patch(patches.Rectangle((-0.200,-0.035),0.40,0.01,\
                facecolor="gray", alpha=0.3 ))
        ax1.add_patch(patches.Rectangle((-0.200,0.025),0.40,0.01,\
                facecolor="gray", alpha=0.3 ))
        ax1.add_patch(patches.Rectangle((0.190,-0.025),0.01,0.05,\
                facecolor="gray", alpha=0.3 ))
        ax1.add_patch(patches.Rectangle((-0.200,-0.025),0.01,0.05,\
                facecolor="gray", alpha=0.3 ))

        ax1.plot(e_x, e_y, lw=1.5)

        ax1.set_title("(Fig. 4.25) Electron Trajectory Starting at (-0.1,-0.01)", weight="bold", size=15)
        ax1.set_xlabel("X [m]", weight="bold", size=11); ax1.set_xlim(xlim)
        ax1.set_ylabel("Y [m]", weight="bold", size=11); ax1.set_ylim(ylim)
        ax1.set_aspect("equal")
        ax1.grid(True)
