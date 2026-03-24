import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from numpy import array, exp, log, sin, cos, sqrt, arccos, arctan
from numpy.linalg import norm
from itertools import product
from scipy.constants import pi, e, m_e as me, k as kb
from scipy.constants import centi as cm, torr, milli, nano
from scipy.interpolate import interp1d
from time import time


## Cross section data of Ar
class ArCrossSections:
    def __init__(self, filename):
        db = np.genfromtxt(filename, delimiter=',', skip_header=3)
        self.sig_el   = interp1d(db[:,0], db[:,1]*1e-20)
        self.sig_ex   = interp1d(db[:,0], db[:,2]*1e-20)
        self.sig_iz   = interp1d(db[:,0], db[:,3]*1e-20)
        self.sig_coll = interp1d(db[:,0], db[:,4]*1e-20)

    def ng(self, p, Tg):
        return p*torr/kb/Tg

    def prob_el(self, x, Te, p, Tg):
        return 1 - exp(-self.ng(p,Tg)*self.sig_el(Te)*x)

    def prob_ex(self, x, Te, p, Tg):
        return 1 - exp(-self.ng(p,Tg)*self.sig_ex(Te)*x)

    def prob_iz(self, x, Te, p, Tg):
        return 1 - exp(-self.ng(p,Tg)*self.sig_iz(Te)*x)

    def prob_coll(self, x, Te, p, Tg):
        return 1 - exp(-self.ng(p,Tg)*self.sig_coll(Te)*x)

    def event(self, Te):
        r = np.random.rand()
        if r>(self.sig_el(Te)+self.sig_ex(Te))/self.sig_coll(Te) and Te>16:
            return 3
        elif r>(self.sig_el(Te))/self.sig_coll(Te) and Te>12:
            return 2
        elif Te>1:
            return 1


## Electric field due to target(chathode)
class Target:
    def __init__(self, voltage, sheath):
        self.vol, self.sh = voltage, sheath
    def Ex(self,x,y,z): return 0
    def Ey(self,x,y,z): return 0
    def Ez(self,x,y,z): return np.minimum(2*self.vol*(self.sh-z)/self.sh**2, 0)
    def __call__(self, p): return np.r_[self.Ex(*p), self.Ey(*p), self.Ez(*p)]


## Magnetic field due to magnets
class Magnet:
    def __init__(self, corner, size, Br=1.4):
        self.x0, self.y0, self.z0 = corner
        self.w, self.l, self.h = size
        self.Br = Br

    def position(self, x, y, z, i, j, k):
        X = x - self.x0 - (i + 1)*self.w/2
        Y = y - self.y0 - (j + 1)*self.l/2
        Z = z - self.z0 - (k + 1)*self.h/2
        return X, Y, Z

    def Bx(self, x, y, z):
        Bx = 0.0
        for i, j, k in product((-1,1), (-1,1), (-1,1)):
            X, Y, Z = self.position(x, y, z, i, j, k)
            Bx = Bx + i*j*k*np.log(Y + np.sqrt(X**2 + Y**2 + Z**2))
        return - Bx*self.Br/4/np.pi

    def By(self, x, y, z):
        By = 0.0
        for i, j, k in product((-1,1), (-1,1), (-1,1)):
            X, Y, Z = self.position(x, y, z, i, j, k)
            By = By + i*j*k*np.log(X + np.sqrt(X**2 + Y**2 + Z**2))
        return - By*self.Br/4/np.pi

    def Bz(self, x, y, z):
        Bz = 0.0
        for i, j, k in product((-1,1), (-1,1), (-1,1)):
            X, Y, Z = self.position(x, y, z, i, j, k)
            Bz = Bz + i*j*k*np.arctan(X*Y/Z/np.sqrt(X**2 + Y**2 + Z**2))
        return Bz*self.Br/4/np.pi

    def __call__(self, p):
        return np.r_[self.Bx(*p), self.By(*p), self.Bz(*p)]


class MagnetPack(Magnet):
    def __init__(self, *magnets):
        self.magnets = magnets
    def Bx(self, x, y, z): return sum([mag.Bx(x,y,z) for mag in self.magnets])
    def By(self, x, y, z): return sum([mag.By(x,y,z) for mag in self.magnets])
    def Bz(self, x, y, z): return sum([mag.Bz(x,y,z) for mag in self.magnets])


## Functions
def update(v1, coll_type=0):
    vx1, vy1, vz1 = v1
    loss = (0, 1, 12, 16)

    ke1   = 0.5*norm(v1)**2*me/e
    ke2   = ke1 - loss[coll_type]
    vmag1 = norm(v1)
    vmag2 = vmag1*sqrt(ke2/ke1)

    theta = arccos(vz1/vmag1)
    phi   = arctan(vy1/vx1)
    rot   = array([[cos(theta)*cos(phi), -sin(phi), sin(theta)*cos(phi)],
                   [cos(theta)*sin(phi), cos(phi), sin(theta)*sin(phi)],
                   [-sin(theta), 0, cos(theta)]])

    r     = np.random.rand()
    chi   = arccos(1 - 2*r/(1+8*ke1*(1-r)/27.21))
    psi   = pi*2*r
    vec   = array([sin(chi)*cos(psi), sin(chi)*sin(psi), cos(chi)])

    return vmag2*rot.dot(vec)


def eqn_motion(u, t, E, B):
    r, v = u[:3], u[3:]
    a = -(E(r) + np.cross(v,B(r)))*e/me
    return np.r_[v, a]


def rk4_solver(f, u0, t, args=()):
    u = array([u0]*t.size, dtype=np.float64)
    for i in range(t.size-1):
        dt = t[i+1] - t[i]
        k1 = dt*f(u[i], t[i], *args)
        k2 = dt*f(u[i]+k1/2., t[i]+dt/2., *args)
        k3 = dt*f(u[i]+k2/2., t[i]+dt/2., *args)
        k4 = dt*f(u[i]+k3, t[i]+dt, *args)
        u[i+1] = u[i] + (k1 + 2*k2 + 2*k3 + k4)/6.
    return u


def move_onestep(dt, p0, v0, E, B):
    u0 = np.r_[p0, v0]
    t  = np.r_[0, dt]
    u  = rk4_solver(eqn_motion, u0, t, args=(E, B))[-1]
    return u[:3], u[3:]


def trace_electron(t, p0, v0, E, B, p, Tg=300):
    result = [np.r_[t[0], p0, v0, 0]]

    p1, v1 = p0, v0
    for i in range(t.size-1):
        coll_type = 0
        dt = t[i+1] - t[i]
        p2, v2 = move_onestep(dt, p1, v1, E, B)

        ke = 0.5*np.linalg.norm(v2)**2*me/e
        pe = -0.5*E.vol*(p2[-1]-E.sh)**2/E.sh**2 if p2[-1]<E.sh else 0.0
        dx = np.linalg.norm(p2-p1)

        if np.random.rand() < ar.prob_coll(dx,ke,p,Tg):
            coll_type = ar.event(ke)
            v2 = update(v2, coll_type)

        result = np.r_[result, [np.r_[t[i+1], p2, v2, coll_type]]]

        if pe+ke <26:
            break
        else:
            p1, v1 = p2, v2

    return result

## Plot results

def show_contour(ax, magnet):
    x = np.linspace(-10*cm, 10*cm, 101)
    y = np.linspace(-3*cm, 3*cm, 101)
    gridx, gridy = np.meshgrid(x, y)
    gridBx = magnet.Bx(gridx, gridy, 0)
    gridBy = magnet.By(gridx, gridy, 0)
    gridBz = magnet.Bz(gridx, gridy, 0)
    normB  = np.sqrt(gridBx**2 + gridBy**2)
    levels = np.linspace(0, 0.16, 9)

    ax.contourf(gridx, gridy, normB, cmap='coolwarm')
    ax.contour(gridx, gridy, gridBz, levels=0, colors='w')
    return ax

def show_magnets(ax):
    ax.add_patch(patches.Rectangle((-0.08,-0.01), 0.16, 0.02,
                            edgecolor='k', facecolor="gray", alpha=0.3 ))
    ax.add_patch(patches.Rectangle((-0.10,-0.02), 0.01, 0.04,
                            edgecolor='k', facecolor="gray", alpha=0.3 ))
    ax.add_patch(patches.Rectangle((-0.10, 0.02), 0.20, 0.01,
                            edgecolor='k', facecolor="gray", alpha=0.3 ))
    ax.add_patch(patches.Rectangle(( 0.09,-0.02), 0.01, 0.04,
                            edgecolor='k', facecolor="gray", alpha=0.3 ))
    ax.add_patch(patches.Rectangle((-0.10,-0.03), 0.20, 0.01,
                            edgecolor='k', facecolor="gray", alpha=0.3 ))
    return ax


##
if __name__ == "__main__":

    ar = ArCrossSections("ArCrossSections.csv")

    target = Target(voltage=-300, sheath=0.001)
    mag1 = Magnet(( -8*cm, -1*cm, -2*cm), (16*cm, 2*cm, 1*cm),  1.4)
    mag2 = Magnet((-10*cm, -2*cm, -2*cm), ( 1*cm, 4*cm, 1*cm), -1.4)
    mag3 = Magnet((-10*cm,  2*cm, -2*cm), (20*cm, 1*cm, 1*cm), -1.4)
    mag4 = Magnet((  9*cm, -2*cm, -2*cm), ( 1*cm, 4*cm, 1*cm), -1.4)
    mag5 = Magnet((-10*cm, -3*cm, -2*cm), (20*cm, 1*cm, 1*cm), -1.4)
    magnet = MagnetPack(mag1, mag2, mag3, mag4, mag5)

    xs = np.linspace(-10*cm, 10*cm, 101)
    ys = np.linspace(-3*cm, 3*cm, 101)

##
    if 1:
        t = np.arange(0, 500, 0.02)*nano
        p0, v0 = (0, -0.007, 0), (0, 0, 0)
        t_start = time()
        x, y, z, vx, vy, vz = rk4_solver(eqn_motion, np.r_[p0, v0], t,
                                    args=(target, magnet)).T
        t_end= time()
        print("Time(s) >> %.2f" % (t_end-t_start))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        show_magnets(ax1)
        show_contour(ax1, magnet)
        ax1.plot(x, y, 'k', lw=1)

        ax1.set_xlim(xs.min(), xs.max())
        ax1.set_ylim(ys.min(), ys.max())
        ax1.set_aspect('equal')

        ke = 0.5*me*(vx**2 + vy**2 + vz**2)/e
        pe = np.zeros_like(ke)
        pe[z < target.sh] = -0.5*target.vol*(z[z < target.sh] - target.sh)**2/target.sh**2
        # ax2.plot(t, ke-pe, 'g', lw=2, label="Total Energy")
        ax2.plot(t, ke, 'k', lw=2, label="Kinetic Energy")
        ax2.plot(t, pe, 'r', lw=2, label="Potential Energy")
        ax2.legend(fontsize=13, frameon=False)
        ax2.grid(c='k', ls=':', lw=1)

        fig.tight_layout()
        plt.show()

##
    if 1:
        np.random.seed(1)
        t = np.arange(0, 500, 0.02)*nano
        p0, v0 = (0, -0.007, 0), (0, 0, 0)
        t_start = time()
        result = trace_electron(t, p0, v0, target, magnet, 5*milli)
        t_end= time()
        print("Time(s) >> %.2f" % (t_end-t_start))
        t, x, y, z, vx, vy, vz, c = result.T

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        show_magnets(ax1)
        show_contour(ax1, magnet)
        ax1.plot(x, y, 'k', lw=1)

        ax1.set_xlim(xs.min(), xs.max())
        ax1.set_ylim(ys.min(), ys.max())
        ax1.set_aspect('equal')

        ke = 0.5*me*(vx**2 + vy**2 + vz**2)/e
        pe = np.zeros_like(ke)
        pe[z < target.sh] = -0.5*target.vol*(z[z < target.sh] - target.sh)**2/target.sh**2
        # ax2.plot(t, ke-pe, 'g', lw=2, label="Total Energy")
        ax2.plot(t, ke, 'g', lw=2, label="Kinetic Energy")
        ax2.plot(t, pe, 'r', lw=2, label="Potential Energy")
        ax2.legend(fontsize=13, frameon=False)
        ax2.grid(c='k', ls=':', lw=1)

        fig.tight_layout()
        plt.show()


##
    if 1:
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,3.5))

        Te = np.logspace(0, 2, 1001)
        ax1.semilogy(Te, ar.sig_el(Te), lw=2, label="elastic")
        ax1.semilogy(Te, ar.sig_ex(Te), lw=2, label="excitation")
        ax1.semilogy(Te, ar.sig_iz(Te), lw=2, label="ionization")
        ax1.set_title("Cross sections(DB)", fontsize=15, fontweight='bold')

        x = np.linspace(0, 1, 1001)
        Te, p, Tg = 100, 5*milli, 300
        ax2.plot(x, ar.prob_el(x,Te,p,Tg), lw=2, label="elastic")
        ax2.plot(x, ar.prob_ex(x,Te,p,Tg), lw=2, label="excitation")
        ax2.plot(x, ar.prob_iz(x,Te,p,Tg), lw=2, label="ionization")
        ax2.set_title("Probability(DB)", fontsize=15, fontweight='bold')

        for ax in (ax1, ax2):
            ax.legend()
            ax.grid()

        fig.tight_layout()
