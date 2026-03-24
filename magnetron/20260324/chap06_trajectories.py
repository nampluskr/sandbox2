from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
from time import clock
import cypct as cy
import matplotlib.patches as patches


########## Global constants for electrons:
E_CHARGE = 1.602E-19 # [C]
E_MASS   = 9.109E-31 # [Kg]
R_CONST  = 8.314
N_A_CONST = 6.022e23
Ptorr = 0.005
Tkelvin = 300


Cross = np.genfromtxt("ArCrossSections.csv", delimiter=',',skip_header=3)
energy = np.linspace(0,1000,10000)
sigEl = np.interp(energy,Cross[:,0],Cross[:,1]*1e-20)
sigEx = np.interp(energy,Cross[:,0],Cross[:,2]*1e-20)
sigIon = np.interp(energy,Cross[:,0],Cross[:,3]*1e-20)
sigTot = np.interp(energy,Cross[:,0],Cross[:,4]*1e-20) 


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
    return np.array([vx,vy,vz,ax,ay,az,0,0])

def solve_rk4(tstep, u0, E, B):
    uu = u0
    h = tstep
    k1 = eval_k(uu, E, B)
    k2 = eval_k(uu + k1*h/2., E, B)
    k3 = eval_k(uu + k2*h/2., E, B)
    k4 = eval_k(uu + k3*h, E, B)
    uuNew = uu + (k1+2*(k2+k3)+k4)*h/6.       
    return uuNew

########## Particle tracing solver using 4th order Runge-Kutta method
class ElectronTrace:
    def __init__(self,t,u0):
        self.t = t
        self.u0 = u0
    def solve(self, E, B):
        self.u = solve_rk4(self.t, self.u0, E, B)
        return self.u
        
 ########## Decide collision event
def decideEvent(KE):
    r = np.random.rand()
    sEl = np.interp(KE,energy,sigEl)
    sEx = np.interp(KE,energy,sigEx)
    TotalSig = np.interp(KE,energy,sigTot)
    if r<sEl/TotalSig and KE>1:
        return 1 #elastic
    elif r<(sEl+sEx)/TotalSig and KE > 12:
        return 2 #excitation
    elif KE>16:
        return 3 #ionization
        
########## Calculate_New_velocity
def newVel(vx,vy,vz,KE,ty):
    vMag = norm(vx,vy,vz)
    if ty == 0:
        vMag2 = vMag*np.sqrt((KE-1)/KE)
    elif ty == 1:
        vMag2 = vMag*np.sqrt((KE-12)/KE)
    else:
        vMag2 = vMag*np.sqrt((KE-6)/KE)
    theta0 = np.arccos(vz/vMag)
    phi0 = np.arctan(np.divide(vx,vy))
    #r = np.random.rand()
    r = 0.3
    psi0 = np.arccos(1-(2*r)/(1+8*(KE/27.12)*(1-r)))
    chi0 = 2*np.pi*r
    
    print theta0, phi0, psi0, chi0, r
    
    MA = np.mat([[np.cos(theta0)*np.cos(phi0), -np.sin(phi0), np.sin(theta0)*np.cos(phi0)],\
    [np.cos(theta0)*np.sin(phi0), np.cos(phi0), np.sin(theta0)*np.sin(phi0)],\
    [-np.sin(theta0), 0, np.cos(theta0)]])
    MB = np.mat([[np.sin(chi0)*np.cos(psi0)],[np.sin(chi0)*np.sin(psi0)],[np.cos(chi0)]])
    return MA*MB*vMag2
        
########## Calculate_Electron_Trejectory
def runE(t,tmax,tstep,u0):
    tj = np.empty([1,8])
    tj = np.delete(tj,0,0)
    while t<tmax:
        pct = ElectronTrace(tstep,u0)
        u = pct.solve(E, mag_array)
        Old_x,Old_y,Old_z = u0[0], u0[1], u0[2]
        New_x,New_y,New_z = u[0], u[1], u[2]
        delx = np.sqrt((Old_x-New_x)**2+(Old_y-New_y)**2+(Old_z-New_z)**2)
        vMag = norm(u[3],u[4],u[5])
        KE = 0.5*E_MASS*vMag**2/E_CHARGE
        prob = 1-np.exp(-Ptorr*133.32/R_CONST/Tkelvin*N_A_CONST*np.interp(KE,energy,sigTot)*delx)
        r = np.random.rand()
        if r < prob:
            ty = decideEvent(KE)
            u[6] = ty
            v2 = newVel(u[3],u[4],u[5],KE,ty)
            u[3],u[4],u[5] = v2[0,0], v2[1,0], v2[2,0]
        else:
            u[6] = 0
        t += tstep # [ns]  Tracing time
        u0 = u
        if u[6] == 3:
            if np.random.rand()<0.1:
                u[7]=1
            tj =np.vstack((tj,u))
    return tj
        

if __name__ == "__main__":

##############################################################################
    if 1: # Test
            ke = 100
            print decideEvent(ke)
    

##############################################################################
    if 0: # Main
        
        
        tstart = clock()
        
        E = cy.ElectricField(-300.0, 0.001, 0.0)
        scale = 1.4/4.0/np.pi
        mag1 = cy.Magnet(-0.175, -0.010, -0.02, 0.35, 0.02, 0.01, scale) # Center
        mag2 = cy.Magnet(-0.200, -0.025, -0.02, 0.01, 0.05, 0.01,-scale) # Bottom
        mag3 = cy.Magnet(-0.200,  0.025, -0.02, 0.40, 0.01, 0.01,-scale) # Right
        mag4 = cy.Magnet( 0.190, -0.025, -0.02, 0.01, 0.05, 0.01,-scale) # Top
        mag5 = cy.Magnet(-0.200, -0.035, -0.02, 0.40, 0.01, 0.01,-scale) # Left
        mag_array = cy.MagneticField(mag1, mag2, mag3, mag4, mag5)

        numE = 1
        max_numE = 1000
        sputterList = np.empty([1,8])
        sputterList = np.delete(sputterList,0,0)
        bigEList = np.empty([1,3])
        bigEList = np.delete(bigEList,0,0)
        n =numE-1
        for i in xrange(500):
            temp_EPt_B = np.array([[np.random.uniform(-0.15,0.15),np.random.uniform(-0.01,-0.025),0.0]])   # [m]   Initial position
            temp_EPt_T = np.array([[np.random.uniform(-0.15,0.15),np.random.uniform(0.01,0.025),0.0]])   # [m]   Initial position
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
                vx0,vy0,vz0 = (0.,0.,0.)         # [m/s] Initial velocity
                u0 = np.array([x0,y0,z0,vx0,vy0,vz0,0,0])#[x0,y0,z0,vx0,vy0,vz0,evt,secE])
                TJ = runE(t,tmax,tstep,u0)
                sputterList = np.vstack((sputterList,TJ))
                numPts = len(sputterList)
                print "Calculate Number%d"%n
                print "Ionization Number%d"%numPts
                n += 1
                SEEQ = np.min([100/len(bigEList),1])
                if np.random.rand()<SEEQ:
                    bigEList = np.vstack((bigEList,sputterList[:,:3]))
        tend = clock()
        print "Time >> %.4e [sec]" % (tend-tstart)
#        print "Total Electrons%d" %len(sputterList)
        
        Target = np.zeros((71,401))
        for k in xrange(numPts):
            i = np.round(sputterList[k,0]/0.001+201)
            j = np.round(sputterList[k,1]/0.001+36)
            if 0<i<=401 and 0<j<=71:
                Target[j-1,i-1] += 1
########################### Erosion top view
        if 0:

            xlim = [-0.205,0.205]
            ylim = [-0.04,0.04]
            fig1 = plt.figure(figsize=(8,6))
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
            
            e_x, e_y, e_z = sputterList[:,0], sputterList[:,1], sputterList[:,2]
            nx,ny = (401,71)
            x = np.linspace(-0.2,0.2,nx)
            y = np.linspace(-0.035,0.035,ny)
            xx,yy = np.meshgrid(x,y)
            ax1.contourf(xx,yy,Target)

            ax1.set_title("(Fig. 1) Ionization position(Top View)", weight="bold", size=15)
            ax1.set_xlabel("X [m]", weight="bold", size=11); ax1.set_xlim(xlim)
            ax1.set_ylabel("Y [m]", weight="bold", size=11); ax1.set_ylim(ylim)
            ax1.set_aspect("equal")
            ax1.grid(True)
########################### Graphic each electrons - top view
        if 0:

            xlim = [-0.205,0.205]
            ylim = [-0.04,0.04]
            fig1 = plt.figure(figsize=(8,6))
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
            
            e_x, e_y, e_z = sputterList[:,0], sputterList[:,1], sputterList[:,2]
            ax1.plot(e_x,e_y,'ro',markersize=3)

            ax1.set_title("(Fig. 1) Ionization position(Top View)", weight="bold", size=15)
            ax1.set_xlabel("X [m]", weight="bold", size=11); ax1.set_xlim(xlim)
            ax1.set_ylabel("Y [m]", weight="bold", size=11); ax1.set_ylim(ylim)
            ax1.set_aspect("equal")
            ax1.grid(True)

########################### Graphic each electrons - Side view
        if 0:
            xlim = [-0.04,0.04]
            ylim = [-0.025,0.01]
            fig2 = plt.figure(figsize=(8,6))
            ax2 = fig2.add_subplot(111)
            plt.hold(True)
            ax2.add_patch(patches.Rectangle((-0.03,-0.02),0.01,0.01,\
                facecolor="gray", alpha=0.3 ))
            ax2.add_patch(patches.Rectangle((0.02,-0.02),0.01,0.01,\
                facecolor="gray", alpha=0.3 ))
            ax2.add_patch(patches.Rectangle((-0.01,-0.02),0.02,0.01,\
                facecolor="gray", alpha=0.3 ))
            ax2.add_patch(patches.Rectangle((-0.03,-0.005),0.06,0.005,\
                facecolor="gray", alpha=0.3 ))
            
            e_x, e_y, e_z = sputterList[:,0], sputterList[:,1], sputterList[:,2]
            ax2.plot(e_y,e_z,'ro',markersize=1)

            ax2.set_title("(Fig. 2) Ionization position(Side View)", weight="bold", size=15)
            ax2.set_xlabel("Y [m]", weight="bold", size=11); ax2.set_xlim(xlim)
            ax2.set_ylabel("Z [m]", weight="bold", size=11); ax2.set_ylim(ylim)
            ax2.set_aspect("equal")
            ax2.grid(True)
            
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
    
