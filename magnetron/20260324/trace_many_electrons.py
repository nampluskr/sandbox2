from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.interpolate import interp1d
from time import clock
import cypct as cy
from numpy import sin, cos

########## Global constants for electrons:
E_CHARGE = 1.602E-19 # [C]
E_MASS   = 9.109E-31 # [Kg]

TIME_MAX_NUM = 100000
TIME_STEP    = 0.01E-9 # [s]

class Plasma_():
    def __init__(self, filename, torr, kelvin):
        data = np.genfromtxt(filename, delimiter=',', skip_header=3)
        self.sig_el = interp1d(data[:,0], data[:,1])
        self.sig_ex = interp1d(data[:,0], data[:,2])
        self.sig_iz = interp1d(data[:,0], data[:,3])
        self.sig_tt = interp1d(data[:,0], data[:,4])
        self.ng = torr*133.32*6.022E23/8.314/kelvin # [#/m^3]

    def prob_el(self, ke, dx):
        return 1-np.exp(-self.ng*self.sig_el(ke)*1.0E-20*dx)
    def prob_ex(self, ke, dx):
        return 1-np.exp(-self.ng*self.sig_ex(ke)*1.0E-20*dx)
    def prob_iz(self, ke, dx):
        return 1-np.exp(-self.ng*self.sig_iz(ke)*1.0E-20*dx)
    def prob_coll(self, ke, dx):
        return 1-np.exp(-self.ng*self.sig_tt(ke)*1.0E-20*dx)
    def type_coll(self, ke):
        sig_el = self.sig_el(ke)
        sig_ex = self.sig_ex(ke)
        sig_tt = self.sig_tt(ke)
        r = np.random.rand()
        if r>(sig_el+sig_ex)/sig_tt and ke>16:
            return 2
        if r>(sig_el)/sig_tt and ke>12:
            return 1
        if ke>1:
            return 0

class Plasma():
    def __init__(self, filename, torr, kelvin):
        data = np.genfromtxt(filename, delimiter=',', skip_header=3)
        self.ke     = data[:,0]        
        self.sig_el = data[:,1]
        self.sig_ex = data[:,2]
        self.sig_iz = data[:,3]
        self.sig_tt = data[:,4]
        self.ng = torr*133.32*6.022E23/8.314/kelvin # [#/m^3]

    def prob_coll(self, ke, dx):
        sig_tt = np.interp(ke,self.ke,self.sig_tt)
        return 1-np.exp(-self.ng*sig_tt*1.0E-20*dx)

    def type_coll(self, ke):
        sig_el = np.interp(ke,self.ke,self.sig_el)
        sig_ex = np.interp(ke,self.ke,self.sig_ex)
        sig_tt = np.interp(ke,self.ke,self.sig_tt)
        r = np.random.rand()
        if r>(sig_el+sig_ex)/sig_tt and ke>16:
            return 2
        if r>(sig_el)/sig_tt and ke>12:
            return 1
        if ke>1:
            return 0

def ke_vec(vx, vy, vz):
    return 0.5*E_MASS*(vx**2 + vy**2 + vz**2)/E_CHARGE

def pe_vec(x, y, z, E):
    return -0.5*(1-np.sign(z-E.sh))*E.vol*(E.sh-z)**2/E.sh**2

def new_velocity(v, coll):
    vx, vy, vz = v
    ke  = cy.kinetic(vx,vy,vz)
    ke2 = ke - [1,12,16][coll]
    vmag  = np.sqrt(vx**2 + vy**2 + vz**2)
    vmag2 = vmag*np.sqrt(ke2/ke)
    
    theta = np.arccos(vx/vmag)
    phi   = np.arctan(vy/vx)
    r     = np.random.rand()
    chi   = np.arccos(1-2*r/(1+8*ke*(1-r)/27.21))
    psi   = np.pi*2*np.random.rand()

    new_vx = vmag2*(cos(theta)*cos(phi)*sin(chi)*cos(psi) \
                   -sin(phi)*sin(chi)*sin(psi) + sin(theta)*cos(phi)*cos(chi))
    new_vy = vmag2*(cos(theta)*sin(phi)*sin(chi)*cos(psi) \
                   +cos(phi)*sin(chi)*sin(psi) + sin(theta)*sin(phi)*cos(chi))
    new_vz = vmag2*(-sin(theta)*sin(chi)*cos(psi) + cos(theta)*cos(chi))

    return np.array([new_vx,new_vy,new_vz])

def move_onestep(p0, v0, tstep, E, B):
    u0 = np.concatenate([p0,v0])
    k1 = cy.eval_k(u0, E, B)
    k2 = cy.eval_k(u0 + k1*tstep/2., E, B)
    k3 = cy.eval_k(u0 + k2*tstep/2., E, B)
    k4 = cy.eval_k(u0 + k3*tstep, E, B)
    u  = u0 + (k1+2*(k2+k3)+k4)*tstep/6.
    return u[:3], u[3:]

def trace_single_electron(p0, v0, tstep, tnum, E, B, torr, kelvin):
    tt = np.empty(tnum);     tt[0] = 0.0
    uu = np.empty((tnum,6)); uu[0] = np.concatenate([p0,v0])
    cc = np.empty(tnum);     cc[0] = -1
    
    ar = Plasma("ArCrossSections.csv", torr, kelvin)
    p1, v1 = p0, v0
    for i in xrange(tnum-1):
        coll = -1
        p2, v2 = move_onestep(p1,v1,tstep,E,B)
        ke = cy.kinetic(v2[0],v2[1],v2[2])
        pe = cy.potential(p2[0],p2[1],p2[2],E)
        dx = distance(p1,p2)
        
        if np.random.rand() < ar.prob_coll(ke,dx):
            coll = ar.type_coll(ke)
            v2 = new_velocity(v2,coll)

        tt[i+1] = tstep*(i+1)
        uu[i+1] = np.concatenate([p2,v2])
        cc[i+1] = coll
        p1, v1 = p2, v2
    
        if pe+ke <27:
            break
            
    return tt[:i+1], uu[:i+1], cc[:i+1]

def trace_many_electrons(pos, vel, E, B, plasma):
    max_num = num_electron = len(pos)
    list_coll_type, list_coll_pos = [], []
   
    for i in xrange(num_electron):
        p1, v1 = pos.pop(), vel.pop()
        num_electron -= 1
        
        for j in xrange(TIME_MAX_NUM - 1):
            p2, v2 = move_onestep(p1,v1,TIME_STEP,E,B)
            ke = cy.kinetic(v2[0],v2[1],v2[2])
            pe = cy.potential(p2[0],p2[1],p2[2],E)
            dx = distance(p1,p2)
        
            if np.random.rand() < plasma.prob_coll(ke,dx):
                coll = plasma.type_coll(ke)
                v2 = new_velocity(v2,coll)
                list_coll_pos.append(p2)
                list_coll_type.append(coll)

            if pe+ke < 16: 
                break
            p1, v1 = p2, v2
            
        print "Electron (%d/%d) -> Collisions: %d" \
            % (i+1, max_num, len(list_coll_type)), j

    return list_coll_pos, list_coll_type

def distance(p1,p2):
    d2 = (p2-p1)**2
    return np.sqrt(d2.sum())

def add_patches(ax,xlim,ylim):
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

    ax.set_title("Electron Trajectory", weight="bold", size=15)
    ax.set_xlabel("X [m]", weight="bold", size=11)
    ax.set_ylabel("Y [m]", weight="bold", size=11)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.grid(True)

if __name__ == "__main__":

##############################################################################
    E = cy.ElectricField(-300, 0.001)
    scale = 1.4/4.0/np.pi
    mag1 = cy.Magnet((-0.08,-0.01,-0.02), (0.16,0.02,0.01), scale) # Center
    mag2 = cy.Magnet((-0.10,-0.02,-0.02), (0.01,0.04,0.01),-scale) # Bottom
    mag3 = cy.Magnet((-0.10, 0.02,-0.02), (0.20,0.01,0.01),-scale) # Right
    mag4 = cy.Magnet(( 0.09,-0.02,-0.02), (0.01,0.04,0.01),-scale) # Top
    mag5 = cy.Magnet((-0.10,-0.03,-0.02), (0.20,0.01,0.01),-scale) # Left
    B = cy.MagneticField(mag1, mag2, mag3, mag4, mag5)
    ar_plasma = Plasma("ArCrossSections.csv", 0.005, 293)
    
    if 1:
        tstart = clock()
        num_electron = 10
        pos, vel= [], []
        for i in xrange(num_electron):
            p = v = np.zeros(3)
            p[0] = np.random.uniform(-0.095,0.095)
            p[1] = np.random.uniform(-0.025,0.025)
            pos.append(p)
            vel.append(v)
   
        p_list, c_list = trace_many_electrons(pos, vel, E, B, ar_plasma)
        uu = np.concatenate(p_list).reshape(-1,3)
        cc = np.array(c_list)

        uu_el  = uu[np.where(cc==0)]
        uu_ex  = uu[np.where(cc==1)]
        uu_iz  = uu[np.where(cc==2)]
        tend = clock()
        print "Time >> %.4e [sec]" % (tend-tstart), cc.size

        fig = plt.figure(figsize=(12,6))
        ax1 = fig.add_subplot(111)
        ax1.plot(uu_el[:,0], uu_el[:,1], 'go', lw=1.5)
        ax1.plot(uu_ex[:,0], uu_ex[:,1], 'bo', lw=1.5)
        ax1.plot(uu_iz[:,0], uu_iz[:,1], 'ro', lw=1.5)
        add_patches(ax1,xlim=[-0.105,0.105],ylim=[-0.035,0.035])

        fig.tight_layout()
        plt.show()    


##############################################################################

