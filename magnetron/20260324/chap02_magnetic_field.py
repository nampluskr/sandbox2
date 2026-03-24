from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

########## Magnetic field [T] (Analytic expression)
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

def norm(x,y,z):
    return np.sqrt(x**2+y**2+z**2)

def f1(x,y,z,h):
    return np.log((norm(x,y,z-h)-y)/(norm(x,y,z-h)+y))

def f2(x,y,z,h):
    return np.arctan((x/y)*(z-h)/norm(x,y,z-h))
    
if __name__ == "__main__":
    
    if 0: 
        scale = 1.4/4.0/np.pi
        mag = Magnet(-0.01/2, -0.01/2, -0.01/2, 0.01, 0.01, 0.01, scale)

        xlim = [-0.05,0.05]
        ylim = [-130, 130]
        x = np.linspace(xlim[0], xlim[1], 100)
        y = mag.Bx(x, 0.0, 0.02)*10000

        fig = plt.figure(figsize=(7,4.5))
        ax = fig.add_subplot(111)
        ax.plot(x, y, lw=2)
        ax.axhline(y=0, c='k', lw=1.5, ls="--")

        ax.set_title("Bx along 2m above the Magnet", weight="bold", size=15)
        ax.set_xlabel("X [m]", weight="bold", size=11); ax.set_xlim(xlim)
        ax.set_ylabel("Bx [G]", weight="bold", size=11); ax.set_ylim(ylim)
        
        ax.grid(True)
        fig.tight_layout()
        plt.show()
        
    if 1:
        scale = 1.4/4.0/np.pi
        mag = Magnet(-0.01/2,-0.01/2,-0.01/2, 0.01, 0.01, 0.01, scale)
        mag_pos = np.array([(-0.005,-0.005),(0.005,-0.005),(0.005,0.005),\
                    (-0.005,0.005),(-0.005,-0.005)])

        ylim = [-0.05,0.05]
        zlim = [-0.05,0.05]
        y = np.linspace(ylim[0], ylim[1], 100)
        z = np.linspace(zlim[0], zlim[1], 100)
        Y, Z = np.meshgrid(y,z)

        meshBy = mag.By(0.0, Y, Z)*10000
        meshBz = mag.Bz(0.0, Y, Z)*10000
        meshB  = np.sqrt(meshBy**2 + meshBz**2)

        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
        ax.plot(mag_pos[:,0], mag_pos[:,1], c='k', lw=2)

        cm = ax.streamplot(Y,Z,meshBy,meshBz, \
		cmap = plt.cm.jet, color = np.clip(meshB,50,500), \
            linewidth=1.5, density=[1,2])
        cb = fig.colorbar(cm.lines, ax=ax, shrink=0.6)
        cb.ax.set_ylabel("|B| [G]", weight="bold", size=11)

        ax.set_title("Magnetic Vector Field", weight="bold", size=15)
        ax.set_xlabel("Y [m]", weight="bold", size=11); ax.set_xlim(ylim)
        ax.set_ylabel("Z [m]", weight="bold", size=11); ax.set_ylim(zlim)

        ax.set_aspect("equal")        
        ax.grid(True)
        fig.tight_layout()
        plt.show()

    if 1:
        scale = 1.4/4.0/np.pi
        mag1 = Magnet(-0.175, -0.010, 0.0, 0.35, 0.02, 0.01, scale)
        mag2 = Magnet(-0.200, -0.025, 0.0, 0.01, 0.05, 0.01,-scale)
        mag3 = Magnet(-0.200,  0.025, 0.0, 0.40, 0.01, 0.01,-scale)
        mag4 = Magnet( 0.190, -0.025, 0.0, 0.01, 0.05, 0.01,-scale)
        mag5 = Magnet(-0.200, -0.035, 0.0, 0.40, 0.01, 0.01,-scale)
        mag_array = MagneticField(mag1, mag2, mag3, mag4, mag5)

        ylim = [-0.065,0.065]
        x = np.linspace(ylim[0], ylim[1], 100)
        y = mag_array.By(0.0,x,0.02)*10000


        fig = plt.figure(figsize=(8,4.5))
        ax = fig.add_subplot(111)
        ax.plot(x,y)
        ax.axhline(y=0, c='k', lw=1.5, ls="--")

        ax.set_title("Parallel Magnetic Field (z=2cm)", weight="bold", size=15)
        #ax.set_xlabel("X [m]", weight="bold", size=11); ax.set_xlim(xlim)
        #ax.set_ylabel("Y [m]", weight="bold", size=11); ax.set_ylim(ylim)

        #ax.set_aspect("equal")                
        ax.grid(True)
        fig.tight_layout()
        plt.show()
        
    if 1:
        scale = 1.4/4.0/np.pi
        mag1 = Magnet(-0.175, -0.010, 0.0, 0.35, 0.02, 0.01, scale) # Center
        mag2 = Magnet(-0.200, -0.025, 0.0, 0.01, 0.05, 0.01,-scale) # Bottom
        mag3 = Magnet(-0.200,  0.025, 0.0, 0.40, 0.01, 0.01,-scale) # Right
        mag4 = Magnet( 0.190, -0.025, 0.0, 0.01, 0.05, 0.01,-scale) # Top
        mag5 = Magnet(-0.200, -0.035, 0.0, 0.40, 0.01, 0.01,-scale) # Left
        mag_array = MagneticField(mag1, mag2, mag3, mag4, mag5)

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

        xlim = [0.05,0.21]
        ylim = [-0.04,0.04]
        x = np.linspace(xlim[0], xlim[1], 500)
        y = np.linspace(ylim[0], ylim[1], 300)
        X, Y = np.meshgrid(x,y)

        meshBx = mag_array.Bx(X, Y, 0.02)*10000
        meshBy = mag_array.By(X, Y, 0.02)*10000
        meshBz = mag_array.Bz(X, Y, 0.02)*10000
        meshB  = np.sqrt(meshBx**2 + meshBy**2)

        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)

        cm = ax.contourf(X,Y,meshB, cmap = plt.cm.jet)
        ax.contourf(X,Y,meshBz, levels=np.linspace(-10,10,3), colors='k')
        ax.streamplot(X,Y,meshBx,meshBy, \
            cmap = plt.cm.jet, color = np.clip(meshB,200,1500), \
            linewidth=1.5, density=1.5)
        
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
        
    if 1:
        scale = 1.4/4.0/np.pi
        mag1 = Magnet(-0.175, -0.010, 0.0, 0.35, 0.02, 0.01, scale) # Center
        mag2 = Magnet(-0.200, -0.025, 0.0, 0.01, 0.05, 0.01,-scale) # Bottom
        mag3 = Magnet(-0.200,  0.025, 0.0, 0.40, 0.01, 0.01,-scale) # Right
        mag4 = Magnet( 0.190, -0.025, 0.0, 0.01, 0.05, 0.01,-scale) # Top
        mag5 = Magnet(-0.200, -0.035, 0.0, 0.40, 0.01, 0.01,-scale) # Left
        mag_array = MagneticField(mag1, mag2, mag3, mag4, mag5)

        xlim = [0.0,0.21]
        ylim = [-0.04,0.04]
        x = np.linspace(xlim[0], xlim[1], 500)
        y = np.linspace(ylim[0], ylim[1], 300)
        X, Y = np.meshgrid(x,y)

        meshBx = mag_array.Bx(X, Y, 0.02)*10000
        meshBy = mag_array.By(X, Y, 0.02)*10000
        meshBz = mag_array.Bz(X, Y, 0.02)*10000
        meshB  = np.sqrt(meshBx**2 + meshBy**2)

        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)

        ax.add_patch(patches.Rectangle((-0.175,-0.010),0.35,0.02,\
                facecolor="gray", alpha=0.3 ))
        ax.add_patch(patches.Rectangle((-0.200,-0.035),0.40,0.01,\
                facecolor="gray", alpha=0.3 ))
        ax.add_patch(patches.Rectangle((-0.200,0.025),0.40,0.01,\
                facecolor="gray", alpha=0.3 ))
        ax.add_patch(patches.Rectangle((0.190,-0.025),0.01,0.05,\
                facecolor="gray", alpha=0.3 ))
        ax.add_patch(patches.Rectangle((-0.200,-0.025),0.01,0.05,\
                facecolor="gray", alpha=0.3 ))
                
        #ax.contourf(X,Y,meshB, cmap = plt.cm.jet)
        ax.contourf(X,Y,meshBz, levels=np.linspace(-10,10,3), colors='k')
        ax.streamplot(X,Y,meshBx,meshBy, \
            cmap = plt.cm.jet, color = np.clip(meshB,200,1500), \
            linewidth=1.5, density=2)

        ax.set_title("Parallel Magnetic Field @z=2cm", weight="bold", size=15)
        ax.set_xlabel("X [m]", weight="bold", size=11); ax.set_xlim(xlim)
        ax.set_ylabel("Y [m]", weight="bold", size=11); ax.set_ylim(ylim)

        ax.set_aspect("equal")                
        ax.grid(True)
        fig.tight_layout()
        plt.show()
        
    if 1:
        scale = 1.4/4.0/np.pi
        mag1 = Magnet(-0.175, -0.010, 0.0, 0.35, 0.02, 0.01, scale) # Center
        mag2 = Magnet(-0.200, -0.025, 0.0, 0.01, 0.05, 0.01,-scale) # Bottom
        mag3 = Magnet(-0.200,  0.025, 0.0, 0.40, 0.01, 0.01,-scale) # Right
        mag4 = Magnet( 0.190, -0.025, 0.0, 0.01, 0.05, 0.01,-scale) # Top
        mag5 = Magnet(-0.200, -0.035, 0.0, 0.40, 0.01, 0.01,-scale) # Left
        mag_array = MagneticField(mag1, mag2, mag3, mag4, mag5)

        ylim = [-0.04,0.04]
        zlim = [0.0,0.04]
        y = np.linspace(ylim[0], ylim[1], 500)
        z = np.linspace(zlim[0], zlim[1], 300)
        Y, Z = np.meshgrid(y,z)

        meshBx = mag_array.Bx(0.0, Y, Z)*10000
        meshBy = mag_array.By(0.0, Y, Z)*10000
        meshBz = mag_array.Bz(0.0, Y, Z)*10000
        meshB  = np.sqrt(meshBy**2 + meshBz**2)

        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)

        ax.add_patch(patches.Rectangle((-0.035,0.0),0.01,0.01,\
                facecolor="gray", alpha=0.3 ))
        ax.add_patch(patches.Rectangle((-0.010,0.0),0.02,0.01,\
                facecolor="gray", alpha=0.3 ))
        ax.add_patch(patches.Rectangle((0.025,0.0),0.01,0.01,\
                facecolor="gray", alpha=0.3 ))
                
        #ax.contourf(Y,Z,meshB, cmap = plt.cm.jet)
        ax.contourf(Y,Z,meshBz, levels=np.linspace(-5,5,3), colors='k')
        ax.streamplot(Y,Z,meshBy,meshBz, \
            cmap = plt.cm.jet, color = np.clip(meshB,200,1500), \
            linewidth=1.5, density=1 )

        ax.set_title("Parallel Magnetic Field @z=2cm", weight="bold", size=15)
        ax.set_xlabel("Y [m]", weight="bold", size=11); ax.set_xlim(ylim)
        ax.set_ylabel("Z [m]", weight="bold", size=11); ax.set_ylim(zlim)

        ax.set_aspect("equal")                
        ax.grid(True)
        fig.tight_layout()
        plt.show()
