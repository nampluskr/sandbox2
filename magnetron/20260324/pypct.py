from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from time import clock
import cypct as cy
import os

########## Global constants for electrons:
E_CHARGE = 1.602E-19 # [C]
E_MASS   = 9.109E-31 # [Kg]

TIME_STEP    = 0.01E-9
TIME_MAX_NUM = 100000

class Plasma():
    def __init__(self, filename, torr, kelvin):
        self.db = np.genfromtxt(filename, delimiter=',', skip_header=3)
        self.ng = torr*133.32*6.022E23/8.314/kelvin # [#/m^3]
    def prob_coll(self, ke, dx):
        sig_tt = np.interp(ke,self.db[:,0],self.db[:,4])
        return 1-np.exp(-self.ng*sig_tt*1.0E-20*dx)
    def prob_iz(self, ke, dx):
        sig_tt = np.interp(ke,self.ke,self.sig_iz)
        return 1-np.exp(-self.ng*sig_tt*1.0E-20*dx)
    def type_coll(self, ke):
        sig_el = np.interp(ke,self.db[:,0],self.db[:,1])
        sig_ex = np.interp(ke,self.db[:,0],self.db[:,2])
        sig_tt = np.interp(ke,self.db[:,0],self.db[:,4])
        r = np.random.rand()
        if r>(sig_el+sig_ex)/sig_tt and ke>16:
            return 2
        if r>(sig_el)/sig_tt and ke>12:
            return 1
        if ke>1:
            return 0

def prob_coll(ke, dx, db, ng):
    sig_tt = np.interp(ke,db[:,0],db[:,4])
    return 1-np.exp(-ng*sig_tt*1.0E-20*dx)

def type_coll(ke, db):
        sig_el = np.interp(ke,db[:,0],db[:,1])
        sig_ex = np.interp(ke,db[:,0],db[:,2])
        sig_tt = np.interp(ke,db[:,0],db[:,4])
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
    return -0.5*(1-np.sign(z-E.sh))*E.vol*(z-E.sh)**2/E.sh**2

def move_onestep(p0, v0, tstep, E, B):
    u0 = np.concatenate([p0,v0])
    k1 = cy.eval_k(u0, E, B)
    k2 = cy.eval_k(u0 + k1*tstep/2., E, B)
    k3 = cy.eval_k(u0 + k2*tstep/2., E, B)
    k4 = cy.eval_k(u0 + k3*tstep, E, B)
    u  = u0 + (k1+2*(k2+k3)+k4)*tstep/6.
    return u[:3], u[3:]

def trace_single_electron(p0, v0, E, B, db, ng):
    p_list, c_list = [], []
    p1, v1 = p0, v0
    
    for i in xrange(TIME_MAX_NUM-1):
        p2, v2 = move_onestep(p1,v1,TIME_STEP,E,B)
        ke = cy.kinetic(v2)
        dx = cy.distance(p1,p2)
        pe = E.potential(p2[0],p2[1],p2[2])
            
        if np.random.rand() < cy.prob_coll(ke,dx,db,ng):
            coll = cy.type_coll(ke,db)
            v2 = cy.new_velocity(v2,coll)
            p_list.append(p2)
            c_list.append(coll)
        
        if ke+pe < 26:
            break
                
        p1, v1 = p2, v2
            
    print ">>> Steps: %5d" % (i+1)
    return p_list, c_list

def trace_many_electrons(p0_list, v0_list, E, B, db, ng):
    max_num = num_electron = len(p0_list)
    pos_list, coll_list = [], []
    
    for i in xrange(num_electron):
        p0, v0 = p0_list.pop(), v0_list.pop()
        num_electron -= 1
        tstart = clock()
        p_list, c_list = trace_single_electron(p0,v0,E,B,db,ng)
        tend = clock()
        print "[%4d/%4d] collisions: %3d / %.2f [s]" % (i+1, max_num, len(c_list), tend-tstart)
        pos_list  += p_list
        coll_list += c_list
            
    return np.concatenate(pos_list).reshape(-1,3), np.array(coll_list)

def draw_magnets(ax,xlim,ylim):
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

def save_result(filename, uu, cc):
    def exist_dir(dirname):
        for root, dirs, files in os.walk(os.getcwd()):
            if dirname in dirs:
                return True
        
    header = "x[m] y[m] z[m] Collision"
    result = np.column_stack((uu,cc))
    if not exist_dir("data"): os.mkdir("data")
    np.savetxt(os.path.join("data",filename), result, \
        fmt="%.4e %.4e %.4e %d", header=header)
    print ">>> %s has been saved." % os.path.join("data",filename)

if __name__ == "__main__":

##############################################################################
### Simulation Input Parameters:
##############################################################################
    TIME_STEP    = 0.01E-9
    TIME_MAX_NUM = 100000

    voltage  = -300
    sheath   = 0.001
    scale    = 0.5
    db_filename = "ArCrossSections.csv"
    torr     = 0.005
    kelvin   = 293
    num_electron = 10
    result_filename = "pct_result_02.txt"
##############################################################################

    E = cy.ElectricField(voltage, sheath)
    mag1 = cy.Magnet(-0.08,-0.01,-0.02, 0.16,0.02,0.01, scale) # Center
    mag2 = cy.Magnet(-0.10,-0.02,-0.02, 0.01,0.04,0.01,-scale) # Bottom
    mag3 = cy.Magnet(-0.10, 0.02,-0.02, 0.20,0.01,0.01,-scale) # Right
    mag4 = cy.Magnet( 0.09,-0.02,-0.02, 0.01,0.04,0.01,-scale) # Top
    mag5 = cy.Magnet(-0.10,-0.03,-0.02, 0.20,0.01,0.01,-scale) # Left
    B = cy.MagneticField([mag1, mag2, mag3, mag4, mag5])
    
    db = np.genfromtxt(db_filename, delimiter=',', skip_header=3)
    ng = torr*133.32*6.022E23/8.314/kelvin
    
    if 1: 
        p0_list, v0_list = [], []
        for i in xrange(num_electron):
            p0 = v0 = np.zeros(3)
            p0[0] = np.random.uniform(-0.095,0.095)
            p0[1] = np.random.uniform(-0.025,0.025)
            p0_list.append(p0)
            v0_list.append(v0)
    
        tstart = clock()
        uu, cc = trace_many_electrons(p0_list,v0_list,E,B,db,ng)
        tend   = clock()
        print "Total time >> %.2f [sec]" % (tend-tstart)
        
        save_result(result_filename, uu, cc)

    if 1:
        data = np.genfromtxt(os.path.join("data",result_filename), skip_header=1)
        uu, cc = data[:,:3], data[:,-1]
        uu_el  = uu[np.where(cc==0)]
        uu_ex  = uu[np.where(cc==1)]
        uu_iz  = uu[np.where(cc==2)]
       
        fig = plt.figure(figsize=(14,6))
        ax1 = plt.subplot2grid((5,6),(0,0),colspan=5,rowspan=3)
        ax2 = plt.subplot2grid((5,6),(0,5),rowspan=3)
        ax3 = plt.subplot2grid((5,6),(3,0),colspan=5,rowspan=2)

        xlim, ylim, zlim = [-0.105,0.105], [-0.035,0.035], [-0.02, 0.02]

        ax1.add_patch(patches.Rectangle((-0.08,-0.01),0.16,0.02,\
            facecolor="gray", alpha=0.3 ))
        ax1.add_patch(patches.Rectangle((-0.10,-0.02),0.01,0.04,\
            facecolor="gray", alpha=0.3 ))
        ax1.add_patch(patches.Rectangle((-0.10, 0.02),0.20,0.01,\
            facecolor="gray", alpha=0.3 ))
        ax1.add_patch(patches.Rectangle(( 0.09,-0.02),0.01,0.04,\
            facecolor="gray", alpha=0.3 ))
        ax1.add_patch(patches.Rectangle((-0.10,-0.03),0.20,0.01,\
            facecolor="gray", alpha=0.3 ))
        ax1.set_xlabel("X [m]", weight="bold", size=11)
        ax1.set_ylabel("Y [m]", weight="bold", size=11)
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim)
        ax1.grid(True)

        ax2.add_patch(patches.Rectangle((-0.02,-0.03),0.01,0.06,\
            facecolor="gray", alpha=0.3 ))
        ax2.set_xlabel("Z [m]", weight="bold", size=11)
        ax2.set_xlim(zlim)
        ax2.set_ylim(ylim)
        ax2.grid(True)

        ax3.add_patch(patches.Rectangle((-0.10,-0.02),0.20,0.01,\
            facecolor="gray", alpha=0.3 ))
        ax3.set_xlabel("X [m]", weight="bold", size=11)
        ax3.set_ylabel("Z [m]", weight="bold", size=11)
        ax3.set_xlim(xlim)
        ax3.set_ylim(zlim)
        ax3.grid(True)
        fig.tight_layout()

        ax1.plot(uu_el[:,0], uu_el[:,1], 'go', lw=0.1)
        ax1.plot(uu_ex[:,0], uu_ex[:,1], 'bo', lw=0.1)
        ax1.plot(uu_iz[:,0], uu_iz[:,1], 'ro', lw=0.1)

        ax2.plot(uu_el[:,2], uu_el[:,1], 'go', lw=0.1)
        ax2.plot(uu_ex[:,2], uu_ex[:,1], 'bo', lw=0.1)
        ax2.plot(uu_iz[:,2], uu_iz[:,1], 'ro', lw=0.1)

        ax3.plot(uu_el[:,0], uu_el[:,2], 'go', lw=0.1)
        ax3.plot(uu_ex[:,0], uu_ex[:,2], 'bo', lw=0.1)
        ax3.plot(uu_iz[:,0], uu_iz[:,2], 'ro', lw=0.1)

        plt.show()
