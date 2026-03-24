from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from time import clock
import matplotlib.patches as patches
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator
import cypct as cy

########## Global constants for electrons:
E_CHARGE = 1.602E-19 # [C]
E_MASS   = 9.109E-31 # [Kg]
R_CONST  = 8.314
N_A_CONST = 6.022e23
Ptorr = 0.005
Tkelvin = 300


Cross = np.genfromtxt("D:\Python\Magnetron_Sputtering_1\ArCrossSections.csv", delimiter=',',skip_header=3)
energy = np.linspace(0,1000,10000)
sigEl = np.interp(energy,Cross[:,0],Cross[:,1]*1e-20)
sigEx = np.interp(energy,Cross[:,0],Cross[:,2]*1e-20)
sigIon = np.interp(energy,Cross[:,0],Cross[:,3]*1e-20)
sigTot = np.interp(energy,Cross[:,0],Cross[:,4]*1e-20) 

########## Import Magnetic field [T] 
tstart = clock()
rawB = np.genfromtxt("D:\Data\Sputter_Erosion\Magnet\Bflux_l_nr.txt", delimiter='',skip_header=9)

nx = 400*4+1
ny = 100*4+1
nz = 20+1

Px = np.linspace(np.min(rawB[:,0]),np.max(rawB[:,0]),nx)
Py = np.linspace(np.min(rawB[:,1]),np.max(rawB[:,1]),ny)
Pz = np.linspace(np.min(rawB[:,2]),np.max(rawB[:,2]),nz)

xi,yi,zi = np.meshgrid(Px,Py,Pz, indexing='ij')
data = griddata(rawB[:,:3],rawB[:,3:6],(xi,yi,zi),method='linear',fill_value=0)

tend = clock()
print "Import Time >> %.4e [sec]" % (tend-tstart)

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
                *self.vol*(self.sh-z)/self.sh**2

########### Magnetic field [T] (Analytic expression)
#class MagneticField:
#    def __init__(self, *magnets):
#        self.mags = magnets
#    def Bx(self,x,y,z):
#        sum_bx = 0.0
#        for m in self.mags:
#            sum_bx += m.Bx(x,y,z)
#        return sum_bx
#    def By(self,x,y,z):
#        sum_by = 0.0
#        for m in self.mags:
#            sum_by += m.By(x,y,z)
#        return sum_by
#    def Bz(self,x,y,z):
#        sum_bz = 0.0
#        for m in self.mags:
#            sum_bz += m.Bz(x,y,z)
#        return sum_bz
#
#class Magnet:
#    def __init__(self,x0,y0,z0,length,width,height,scale):
#        self.a, self.b, self.c  = length, width, height
#        self.x0,self.y0,self.z0 = x0, y0, z0
#        self.scale = scale
#    def Bx(self,x,y,z):
#        x, y, z = x-self.x0, y-self.y0, z-self.z0
#        bx = f1(self.a-x,y,z,self.c) + f1(self.a-x,self.b-y,z,self.c) \
#           - f1(x,y,z,self.c)        - f1(x,self.b-y,z,self.c) \
#           - f1(self.a-x,y,z,0)      - f1(self.a-x,self.b-y,z,0) \
#           + f1(x,y,z,0)             + f1(x,self.b-y,z,0)
#        return -self.scale*bx*0.5
#    def By(self,x,y,z):
#        x, y, z = x-self.x0, y-self.y0, z-self.z0
#        by = f1(self.b-y,x,z,self.c) + f1(self.b-y,self.a-x,z,self.c) \
#           - f1(y,x,z,self.c)        - f1(y,self.a-x,z,self.c) \
#           - f1(self.b-y,x,z,0)      - f1(self.b-y,self.a-x,z,0) \
#           + f1(y,x,z,0)             + f1(y,self.a-x,z,0)
#        return -self.scale*by*0.5
#    def Bz(self,x,y,z):
#        x, y, z = x-self.x0, y-self.y0, z-self.z0
#        bz = f2(y,self.a-x,z,self.c) + f2(self.b-y,self.a-x,z,self.c) \
#           + f2(x,self.b-y,z,self.c) + f2(self.a-x,self.b-y,z,self.c) \
#           + f2(self.b-y,x,z,self.c) + f2(y,x,z,self.c) \
#           + f2(self.a-x,y,z,self.c) + f2(x,y,z,self.c) \
#           - f2(y,self.a-x,z,0)      - f2(self.b-y,self.a-x,z,0) \
#           - f2(x,self.b-y,z,0)      - f2(self.a-x,self.b-y,z,0) \
#           - f2(self.b-y,x,z,0)      - f2(y,x,z,0) \
#           - f2(self.a-x,y,z,0)      - f2(x,y,z,0)
#        return -self.scale*bz


########## Viewfactor
class ViewfactorCalculate:
    def __init__(self, targPts, subPts,targArea,subArea,targ_emi):
        self.targPts = targPts
        self.subPts = subPts
        self.targArea = targArea
        self.subArea = subArea
        self.targ_emi = targ_emi.flatten()
    def f12(self,subIndex):
        diffVect = np.array([self.targPts[0]-self.subPts[0,subIndex],\
        self.targPts[1]-self.subPts[1,subIndex],\
        self.targPts[2]-self.subPts[2,subIndex]])
        dot1 = normal(diffVect[0]*0,diffVect[1]*0,diffVect[2]*1)
        dot2 = normal(diffVect[0]*0,diffVect[1]*0,diffVect[2]*-1)
        tmp = (-dot1[0]*dot2[0]+-dot1[1]*dot2[1]+-dot1[2]*dot2[2])/np.pi/norm(diffVect[0],diffVect[1],diffVect[2])**2
        return np.maximum(tmp,0)
    def depRate(self,subIndex):
        cellVFs = self.f12(subIndex)
        return np.sum(cellVFs*self.targ_emi*self.targArea)/self.subArea

def norm(x,y,z):
    return np.sqrt(x**2+y**2+z**2)
    
def normal(x,y,z):
    return np.array([x/norm(x,y,z),y/norm(x,y,z),z/norm(x,y,z)])

def f1(x,y,z,h):
    return np.log((norm(x,y,z-h)-y)/(norm(x,y,z-h)+y))

def f2(x,y,z,h):
    return np.arctan((x/y)*(z-h)/norm(x,y,z-h))

def eval_k(u, E, B):
    E_CH_M = E_CHARGE/E_MASS
    ax = -(E.Ex(u[0],u[1],u[2]) + u[4]*B[0,2] \
         - u[5]*B[0,1])*E_CH_M
    ay = -(E.Ey(u[0],u[1],u[2]) + u[5]*B[0,0] \
         - u[3]*B[0,2])*E_CH_M
    az = -(E.Ez(u[0],u[1],u[2]) + u[3]*B[0,1] \
         - u[4]*B[0,0])*E_CH_M
    return np.array([u[3],u[4],u[5],ax,ay,az,0,0])

def solve_rk4(tstep, u0, E, B):
    h = tstep
    interp_B = B(u0[:3])
    k1 = eval_k(u0, E, interp_B)
    k2 = eval_k(u0 + k1*h/2., E, interp_B)
    k3 = eval_k(u0 + k2*h/2., E, interp_B)
    k4 = eval_k(u0 + k3*h, E, interp_B)
    uNew = u0 + (k1+2*(k2+k3)+k4)*h/6
    return uNew

########## Particle tracing solver using 4th order Runge-Kutta method
class ElectronTrace:
    def __init__(self,t,u0):
        self.t = t
        self.u0 = u0
    def solve(self, E, B):
        self.u = solve_rk4(self.t, self.u0, E, B)
        return self.u
        
########## Calculate_Collision
class Calculate_Collision:
    def __init__(self,t,tmax,tstep,u0):
        self.t = t
        self.tmax = tmax
        self.tstep = tstep
        self.u0 = u0
    def runE(self,E,B):
        tj = np.empty([1,8])
        tj = np.delete(tj,0,0)
        while self.t<self.tmax and self.totEnergy()>26:
            pct = ElectronTrace(self.tstep,self.u0)
            u = pct.solve(E, B)
            delx = np.sqrt((self.u0[0]-u[0])**2+(self.u0[1]-u[1])**2+(self.u0[2]-u[2])**2)
            self.vMag = norm(u[3], u[4], u[5])
            self.KE = 0.5*E_MASS*self.vMag**2/E_CHARGE
            prob = 1-np.exp(-Ptorr*133.32/R_CONST/Tkelvin*N_A_CONST*np.interp(self.KE,energy,sigTot)*delx)
            if np.random.rand() < prob:
                self.ty = self.decideEvent()
                u[6] = self.ty
#                v2 = self.newVel(u)
#                u[3],u[4],u[5] = v2[0], v2[1], v2[2]
            else:
                u[6] = 0
            self.t += self.tstep # [ns]  Tracing time
            self.u0 = u
            tj = np.vstack((tj,u))
        return tj
    def runE_Ion(self,E,B):
        tj = np.empty([1,8])
        tj = np.delete(tj,0,0)
        while self.t<self.tmax and self.totEnergy()>26:
            pct = ElectronTrace(self.tstep,self.u0)
            u = pct.solve(E, B)
            delx = np.sqrt((self.u0[0]-u[0])**2+(self.u0[1]-u[1])**2+(self.u0[2]-u[2])**2)
            self.vMag = norm(u[3], u[4], u[5])
            self.KE = 0.5*E_MASS*self.vMag**2/E_CHARGE
            prob = 1-np.exp(-Ptorr*133.32/R_CONST/Tkelvin*N_A_CONST*np.interp(self.KE,energy,sigTot)*delx)
            if np.random.rand() < prob:
                self.ty = self.decideEvent()
                u[6] = self.ty
                v2 = self.newVel(u)
                u[3],u[4],u[5] = v2[0], v2[1], v2[2]
            else:
                u[6] = 0
            u[7] = 0.5*E_MASS*self.vMag**2/E_CHARGE
            self.t += self.tstep # [ns]  Tracing time
            self.u0 = u
            if u[6] == 3:
#                if np.random.rand()<0.1:
#                    u[7]=1
                tj = np.vstack((tj,u))
        return tj
    def totEnergy(self):
        return 0.5*E.Ez(self.u0[0],self.u0[1],self.u0[2])*np.minimum(self.u0[2]-0.001,0)\
        +0.5*E_MASS*norm(self.u0[3],self.u0[4],self.u0[5])**2/E_CHARGE
    def newVel(self,u):
        if self.ty == 0:
            vMag2 = self.vMag*np.sqrt((self.KE)/self.KE)
        elif self.ty == 1:
            vMag2 = self.vMag*np.sqrt((self.KE-12)/self.KE)
        else:
            vMag2 = self.vMag*np.sqrt((self.KE-16+10)/self.KE)
        theta0 = np.arccos(u[5]/self.vMag)
        phi0 = np.arctan(np.divide(u[3],u[4]))
        r = np.random.rand()
        psi0 = np.arccos(1-(2*r)/(1+8*(self.KE/27.12)*(1-r)))
        chi0 = 2*np.pi*r 
        MA = np.mat([[np.cos(theta0)*np.cos(phi0), -np.sin(phi0), np.sin(theta0)*np.cos(phi0)],\
        [np.cos(theta0)*np.sin(phi0), np.cos(phi0), np.sin(theta0)*np.sin(phi0)],\
        [-np.sin(theta0), 0, np.cos(theta0)]])
        MB = np.mat([[np.sin(chi0)*np.cos(psi0)],[np.sin(chi0)*np.sin(psi0)],[np.cos(chi0)]])
        return MA*MB*vMag2
    def decideEvent(self):
        r = np.random.rand()
        sEl = np.interp(self.KE,energy,sigEl)
        sEx = np.interp(self.KE,energy,sigEx)
        TotalSig = np.interp(self.KE,energy,sigTot)
        if r<sEl/TotalSig and self.KE>1:
            return 1 #elastic
        elif r<(sEl+sEx)/TotalSig and self.KE > 12:
            return 2 #excitation
        elif self.KE>16:
            return 3 #ionization


if __name__ == "__main__":

##############################################################################
    if 1: # Main
        
        
        tstart = clock()
        
        E = cy.ElectricField(-300.0, 0.001, 0.0)
#        scale = 1.4/4.0/np.pi
#        mag1 = cy.Magnet(-0.175, -0.010, -0.02, 0.35, 0.02, 0.01, scale) # Center
#        mag2 = cy.Magnet(-0.200, -0.025, -0.02, 0.01, 0.05, 0.01,-scale) # Bottom
#        mag3 = cy.Magnet(-0.200,  0.025, -0.02, 0.40, 0.01, 0.01,-scale) # Right
#        mag4 = cy.Magnet( 0.190, -0.025, -0.02, 0.01, 0.05, 0.01,-scale) # Top
#        mag5 = cy.Magnet(-0.200, -0.035, -0.02, 0.40, 0.01, 0.01,-scale) # Left
#        mag_array = cy.MagneticField(mag1, mag2, mag3, mag4, mag5)
        B = RegularGridInterpolator((Px,Py,Pz),data)

        numE = 1
        max_numE = 1
        sputterList = np.empty([1,8])
        sputterList = np.delete(sputterList,0,0)
        bigEList = np.empty([1,3])
        bigEList = np.delete(bigEList,0,0)
        n =numE-1
        for i in xrange(1):
            temp_EPt_B = np.array([np.random.uniform(-0.15,0.15),np.random.uniform(-0.01,-0.025),0.0])   # [m]   Initial position
            temp_EPt_T = np.array([np.random.uniform(-0.15,0.15),np.random.uniform(0.01,0.025),0.0])   # [m]   Initial position
            bigEList = np.vstack((bigEList,temp_EPt_B))
            bigEList = np.vstack((bigEList,temp_EPt_T))
        while n < max_numE:
            if len(bigEList)<1:
                break
            else:
                t = 0
                tmax = 260e-9
                tstep = 1e-11
                x0, y0, z0  = bigEList[0][0],bigEList[0][1], 0   # [m]   Initial position
                bigEList = np.delete(bigEList,0,0)
                vx0,vy0,vz0 = (0,0,0)         # [m/s] Initial velocity
                u0 = np.array([x0,y0,z0,vx0,vy0,vz0,0,0])#[x0,y0,z0,vx0,vy0,vz0,evt,secE])
                Cal_col = Calculate_Collision(t,tmax,tstep,u0)
                TJ = Cal_col.runE(E,B)
                sputterList = np.vstack((sputterList,TJ))
                numPts = len(sputterList)
                print "Calculate Number%d"%n
#                print "Ionization Number%d"%numPts
                n += 1
                if np.random.rand()<np.minimum(np.divide(100,len(bigEList)),1):
                    bigEList = np.vstack((bigEList,sputterList[:,:3]))
#                elif np.divide(100,len(bigEList)) == 0:
#                    bigEList = np.vstack((bigEList,sputterList[:,:3]))
        tend = clock()
        print "Time >> %.4e [sec]" % (tend-tstart)
        print "Total Electrons%d" %len(sputterList)
        
#        Top_Ero = np.zeros((71,401))
#        for k in xrange(numPts):
#            i = np.round(sputterList[k,0]/0.001+201)
#            j = np.round(sputterList[k,1]/0.001+36)
#            if 0<i<=401 and 0<j<=71:
#                Top_Ero[j-1,i-1] += 1
#        Side_Ero = np.sum(Top_Ero,axis=1)
            
########################### Deporition top view
        if 0:
            tar_nx,tar_ny = (401,71)
            tar_x = np.linspace(-0.2,0.2,tar_nx)
            tar_y = np.linspace(-0.035,0.035,tar_ny)
            tar_xx,tar_yy = np.meshgrid(tar_x,tar_y)
            targ_x,targ_y = tar_xx.flatten(), tar_yy.flatten()
            targ_z = np.zeros(np.shape(targ_x))
            targ_emi = Top_Ero
            targPts = np.vstack((targ_x,targ_y,targ_z))
    
            sub_nx,sub_ny = (401, 71)
            sub_x = np.linspace(-0.2,0.2,sub_nx)
            sub_y = np.linspace(-0.035,0.035,sub_ny)
            sub_xx,sub_yy = np.meshgrid(sub_x,sub_y)
            subs_x,subs_y = sub_xx.flatten(), sub_yy.flatten()
            subs_z = np.ones(np.shape(subs_x))*0.1
            subPts = np.vstack((subs_x,subs_y,subs_z))
         
            vfCalc = ViewfactorCalculate(targPts,subPts,0.0001,0.0001,targ_emi)
            depRates = np.zeros(np.size(subPts,1))
            for i in xrange(np.size(subPts,1)):
                depRates[i] = vfCalc.depRate(i)
#                print i
            depRates = np.reshape(depRates,np.shape(sub_xx))
            
            xlim = [-0.205,0.205]
            ylim = [-0.04,0.04]
            fig1 = plt.figure(figsize=(8*2,6*2))
            ax1 = fig1.add_subplot(111)
            plt.hold(True)
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
            
            ax1.contourf(sub_x,sub_y,depRates)
            
            ax1.set_title("Deposition (Top View)", weight="bold", size=15)
            ax1.set_xlabel("X [m]", weight="bold", size=11); ax1.set_xlim(xlim)
            ax1.set_ylabel("Y [m]", weight="bold", size=11); ax1.set_ylim(ylim)
            ax1.set_aspect("equal")
            ax1.grid(True)
########################### Erosion top view
        if 0:

            xlim = [-0.205,0.205]
            ylim = [-0.04,0.04]
            fig2 = plt.figure(figsize=(8*2,6*2))
            ax2 = fig2.add_subplot(111)
            plt.hold(True)
            ax2.add_patch(patches.Rectangle((-0.175,-0.010),0.35,0.02,\
            facecolor="gray", alpha=0.3 ))
            ax2.add_patch(patches.Rectangle((-0.200,-0.035),0.40,0.01,\
            facecolor="gray", alpha=0.3 ))
            ax2.add_patch(patches.Rectangle((-0.200,0.025),0.40,0.01,\
            facecolor="gray", alpha=0.3 ))
            ax2.add_patch(patches.Rectangle((0.190,-0.025),0.01,0.05,\
            facecolor="gray", alpha=0.3 ))
            ax2.add_patch(patches.Rectangle((-0.200,-0.025),0.01,0.05,\
            facecolor="gray", alpha=0.3 ))
            
            e_x, e_y, e_z = sputterList[:,0], sputterList[:,1], sputterList[:,2]
            nx,ny = (401,71)
            x = np.linspace(-0.2,0.2,nx)
            y = np.linspace(-0.035,0.035,ny)
            xx,yy = np.meshgrid(x,y)
            ax2.contourf(xx,yy,Top_Ero, cmp = plt.cm.rainbow)
#            ax1.surface(xx,yy,Top_Ero, cmp = plt.cm.rainbow)

            ax2.set_title("Erosion position(Top View)", weight="bold", size=15)
            ax2.set_xlabel("X [m]", weight="bold", size=11); ax2.set_xlim(xlim)
            ax2.set_ylabel("Y [m]", weight="bold", size=11); ax2.set_ylim(ylim)
            ax2.set_aspect("equal")
            ax2.grid(True)
########################### Graphic each electrons - top view
        if 1:

            xlim = [-0.205,0.205]
            ylim = [-0.04,0.04]
            fig3 = plt.figure(figsize=(8*2,6*2))
            ax3 = fig3.add_subplot(111)
            plt.hold(True)
            ax3.add_patch(patches.Rectangle((-0.175,-0.010),0.35,0.02,\
            facecolor="gray", alpha=0.3 ))
            ax3.add_patch(patches.Rectangle((-0.200,-0.035),0.40,0.01,\
            facecolor="gray", alpha=0.3 ))
            ax3.add_patch(patches.Rectangle((-0.200,0.025),0.40,0.01,\
            facecolor="gray", alpha=0.3 ))
            ax3.add_patch(patches.Rectangle((0.190,-0.025),0.01,0.05,\
            facecolor="gray", alpha=0.3 ))
            ax3.add_patch(patches.Rectangle((-0.200,-0.025),0.01,0.05,\
            facecolor="gray", alpha=0.3 ))
            
            e_x, e_y, e_z = sputterList[:,0], sputterList[:,1], sputterList[:,2]
#            ax3.plot(e_x,e_y,'ro',markersize=1)
            ax3.plot(e_x,e_y)

            ax3.set_title("(Fig. 1) Ionization position(Top View)", weight="bold", size=15)
            ax3.set_xlabel("X [m]", weight="bold", size=11); ax3.set_xlim(xlim)
            ax3.set_ylabel("Y [m]", weight="bold", size=11); ax3.set_ylim(ylim)
            ax3.set_aspect("equal")
            ax3.grid(True)

########################### Graphic each electrons - Side view
        if 0:
            xlim = [-0.04,0.04]
            ylim = [-0.025,0.01]
            fig4 = plt.figure(figsize=(8*2,6*2))
            ax4 = fig4.add_subplot(111)
            plt.hold(True)
            ax4.add_patch(patches.Rectangle((-0.035,-0.02),0.01,0.01,\
                facecolor="gray", alpha=0.3 ))
            ax4.add_patch(patches.Rectangle((0.025,-0.02),0.01,0.01,\
                facecolor="gray", alpha=0.3 ))
            ax4.add_patch(patches.Rectangle((-0.01,-0.02),0.02,0.01,\
                facecolor="gray", alpha=0.3 ))
            ax4.add_patch(patches.Rectangle((-0.035,-0.005),0.07,0.005,\
                facecolor="gray", alpha=0.3 ))
            
            e_x, e_y, e_z = sputterList[:,0], sputterList[:,1], sputterList[:,2]
            ax4.plot(e_y,e_z,'ro',markersize=1)

            ax4.set_title("(Fig. 2) Ionization position(Side View)", weight="bold", size=15)
            ax4.set_xlabel("Y [m]", weight="bold", size=11); ax4.set_xlim(xlim)
            ax4.set_ylabel("Z [m]", weight="bold", size=11); ax4.set_ylim(ylim)
            ax4.set_aspect("equal")
            ax4.grid(True)
            
############################# Save images file            
        if 0:
            fig2 = plt.figure(figsize=(8,6))
            ax2 = fig2.add_subplot(111)
    
            xlim = [-0.205,0]
            ylim = [-0.04,0.04]

            ax2.add_patch(patches.Rectangle((-0.175,-0.010),0.35,0.02,\
                    facecolor="gray", alpha=0.3 ))
            ax2.add_patch(patches.Rectangle((-0.200,-0.035),0.40,0.01,\
                    facecolor="gray", alpha=0.3 ))
            ax2.add_patch(patches.Rectangle((-0.200,0.025),0.40,0.01,\
                    facecolor="gray", alpha=0.3 ))
            ax2.add_patch(patches.Rectangle((0.190,-0.025),0.01,0.05,\
                    facecolor="gray", alpha=0.3 ))
            ax2.add_patch(patches.Rectangle((-0.200,-0.025),0.01,0.05,\
                    facecolor="gray", alpha=0.3 ))
            ax2.set_title("(Fig. 1) Electron Trajectory Starting at (-0.1,-0.01)", weight="bold", size=15)
            ax2.set_xlabel("X [m]", weight="bold", size=11); ax2.set_xlim(xlim)
            ax2.set_ylabel("Y [m]", weight="bold", size=11); ax2.set_ylim(ylim)
            ax2.set_aspect("equal")
            ax2.grid(True)
            
            ani_step = 500
            for i in range(1,len(e_x),ani_step):
                ax2.plot(e_x[i],e_y[i],'bo', markersize=1)
                ax2.plot(e_x[i-ani_step:i],e_y[i-ani_step:i],'b')
                filename="test%s.png" % str(i).zfill(4)
                savefig(filename)
#####################################################
        if 1:
            mag1_pos = np.array([(-0.175,-0.005),(0.175,-0.005),(0.175,0.005),\
                    (-0.175,0.005),(-0.175,-0.005)])
            mag2_pos = np.array([(-0.200,-0.035),(0.200,-0.035),(0.200,-0.025),\
                    (-0.200,-0.025),(-0.200,-0.035)])
            mag3_pos = np.array([(0.190,-0.025),(0.200,-0.025),(0.200,0.025),\
                    (0.190,0.025),(0.190,-0.025)])
            mag4_pos = np.array([(-0.200,0.035),(0.200,0.035),(0.200,0.025),\
                    (-0.200,0.025),(-0.200,0.035)])
            mag5_pos = np.array([(-0.190,-0.025),(-0.200,-0.025),(-0.200,0.025),\
                    (-0.190,0.025),(-0.190,-0.025)])

            xlim = [-0.205,0.205]
            ylim = [-0.04,0.04]
            x = np.linspace(xlim[0], xlim[1], 500/8)
            y = np.linspace(ylim[0], ylim[1], 300/8)
            z = 0.02
            X, Y, Z = np.meshgrid(x,y,z,indexing='ij')
#                X, Y = np.meshgrid(x,y, indexing='ij')
                
#                rawB = np.genfromtxt("D:\Data\Sputter_Erosion\Magnet\Bflux_l_nr.txt", delimiter='',skip_header=9	)
            tstart = clock()
            meshBxyz = griddata(rawB[:,:3],rawB[:,3:6],(X,Y,Z),method='linear',fill_value=0)
#                meshBxy = griddata(rawB[:,:2],rawB[:,3:6],(X,Y),method='cubic')*10000
            tend = clock()
            print "Griddata Time >> %.4e [sec]" % (tend-tstart)
#                meshBx = meshBxy[:,:,0]
#                meshBy = meshBxy[:,:,1]
#                meshBz = meshBxy[:,:,2]
            meshBx = meshBxyz[:,:,0,0]*1000
            meshBy = meshBxyz[:,:,0,1]*1000
            meshBz = meshBxyz[:,:,0,2]*1000
            meshB  = np.sqrt(meshBx**2 + meshBy**2)

            fig = plt.figure(figsize=(8*2,6*2))
            ax = fig.add_subplot(111)
                
            X = X[:,:,0]
            Y = Y[:,:,0]
#                X = X[:,:]
#                Y = Y[:,:]
            cm = ax.contourf(X,Y,meshB, cmap = plt.cm.jet)
#                ax.contourf(X,Y,meshBz, levels=np.linspace(-10,10,3), colors='k')
#                ax.streamplot(X,Y,meshBx,meshBy, \
#                cmap = plt.cm.jet, color = np.clip(meshB,200,1500), \
#                linewidth=1.5, density=1.5)
        
            ax.plot(mag1_pos[:,0], mag1_pos[:,1], c='k', lw=1.2, ls="--")
            ax.plot(mag2_pos[:,0], mag2_pos[:,1], c='k', lw=1.2, ls="--")
            ax.plot(mag3_pos[:,0], mag3_pos[:,1], c='k', lw=1.2, ls="--")
            ax.plot(mag4_pos[:,0], mag4_pos[:,1], c='k', lw=1.2, ls="--")
            ax.plot(mag5_pos[:,0], mag5_pos[:,1], c='k', lw=1.2, ls="--")

            ax.set_title("Parallel Magnetic Field @z=2cm", weight="bold", size=15)
            ax.set_xlabel("X [m]", weight="bold", size=11); ax.set_xlim(xlim)
            ax.set_ylabel("Y [m]", weight="bold", size=11); ax.set_ylim(ylim)

            ax.set_aspect("equal")                
            ax.grid(True)
            fig.tight_layout()
            plt.show()
