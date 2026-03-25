# cython: nonecheck = True
# cython: boundscheck = False
# cython: wraparound  = False
# cython: cdivision = True
import numpy as np
cimport numpy as np
from libc.math cimport log, sqrt, atan, acos, sin, cos, pow, exp

########## Global constants for electrons:
DEF E_CHARGE = 1.602E-19 # [C]
DEF E_MASS   = 9.109E-31 # [Kg]
DEF PI       = 3.141592

########## Electric Field [V/m]
cdef class ElectricField:
    cdef double vol, sh
    def __cinit__(self, double voltage, double sheath):
        self.vol = voltage
        self.sh = sheath
    cdef inline double x(self, double x, double y, double z):
        return 0.0
    cdef inline double y(self, double x, double y, double z):
        return 0.0
    cdef inline double z(self, double x, double y, double z):
        if z < self.sh:
            return 2.0*self.vol*(self.sh-z)/self.sh**2
        else:
            return 0.0

########## Magnetic field [T] (Analytic expression)
cdef class MagneticField:
    cdef list mags
    def __cinit__(self, list magnets):
        self.mags = magnets
    cdef double x(self, double x, double y, double z):
        cdef double sum_bx = 0.0
        cdef Magnet m
        for m in self.mags:
            sum_bx += m.x(x,y,z)
        return sum_bx
    cdef double y(self, double x, double y, double z):
        cdef double sum_by = 0.0
        cdef Magnet m
        for m in self.mags:
            sum_by += m.y(x,y,z)
        return sum_by
    cdef double z(self, double x, double y, double z):
        cdef double sum_bz = 0.0
        cdef Magnet m
        for m in self.mags:
            sum_bz += m.z(x,y,z)
        return sum_bz

cdef class Magnet:
    cdef double x0, y0, z0, a, b, c, scale
    def __cinit__(self, double x0, double y0, double z0,
            double a, double b, double c, double scale):
        self.x0,self.y0,self.z0 = x0, y0, z0
        self.a, self.b, self.c  = a, b, c
        self.scale = scale
    cdef double x(self, double _x, double _y, double _z):
        cdef double x = _x-self.x0, y = _y-self.y0, z = _z-self.z0, bx
        bx = f1(self.a-x,y,z,self.c) + f1(self.a-x,self.b-y,z,self.c) \
           - f1(x,y,z,self.c)        - f1(x,self.b-y,z,self.c) \
           - f1(self.a-x,y,z,0)      - f1(self.a-x,self.b-y,z,0) \
           + f1(x,y,z,0)             + f1(x,self.b-y,z,0)
        return -self.scale*bx*0.5*1.4/4.0/PI
    cdef double y(self, double _x, double _y, double _z):
        cdef double x = _x-self.x0, y = _y-self.y0, z = _z-self.z0, by
        by = f1(self.b-y,x,z,self.c) + f1(self.b-y,self.a-x,z,self.c) \
           - f1(y,x,z,self.c)        - f1(y,self.a-x,z,self.c) \
           - f1(self.b-y,x,z,0)      - f1(self.b-y,self.a-x,z,0) \
           + f1(y,x,z,0)             + f1(y,self.a-x,z,0)
        return -self.scale*by*0.5*1.4/4.0/PI
    cdef double z(self, double _x, double _y, double _z):
        cdef double x = _x-self.x0, y = _y-self.y0, z = _z-self.z0, bz
        bz = f2(y,self.a-x,z,self.c) + f2(self.b-y,self.a-x,z,self.c) \
           + f2(x,self.b-y,z,self.c) + f2(self.a-x,self.b-y,z,self.c) \
           + f2(self.b-y,x,z,self.c) + f2(y,x,z,self.c) \
           + f2(self.a-x,y,z,self.c) + f2(x,y,z,self.c) \
           - f2(y,self.a-x,z,0)      - f2(self.b-y,self.a-x,z,0) \
           - f2(x,self.b-y,z,0)      - f2(self.a-x,self.b-y,z,0) \
           - f2(self.b-y,x,z,0)      - f2(y,x,z,0) \
           - f2(self.a-x,y,z,0)      - f2(x,y,z,0)
        return -self.scale*bz*1.4/4.0/PI

cdef inline double norm(double x, double y, double z):
    return sqrt(x*x + y*y + z*z)

cdef inline double f1(double x, double y, double z, double h):
    return log((norm(x,y,z-h)-y)/(norm(x,y,z-h)+y))

cdef inline double f2(double x, double y, double z, double h):
    return atan((x/y)*(z-h)/norm(x,y,z-h))

##########################################################################
cdef double[::1] eval_k(double[::1] u, ElectricField E, MagneticField B):
    cdef double[::1] k = np.empty(6)
    k[0] = u[3]
    k[1] = u[4]
    k[2] = u[5]
    k[3] = -(E.x(u[0],u[1],u[2]) + u[4]*B.z(u[0],u[1],u[2]) \
         - u[5]*B.y(u[0],u[1],u[2]))*(E_CHARGE/E_MASS)
    k[4] = -(E.y(u[0],u[1],u[2]) + u[5]*B.x(u[0],u[1],u[2]) \
         - u[3]*B.z(u[0],u[1],u[2]))*(E_CHARGE/E_MASS)
    k[5] = -(E.z(u[0],u[1],u[2]) + u[3]*B.y(u[0],u[1],u[2]) \
         - u[4]*B.x(u[0],u[1],u[2]))*(E_CHARGE/E_MASS)
    return k

cdef double[::1] move_onestep(double[::1] pv0, double tstep, \
        ElectricField E, MagneticField B):
    cdef:
        double[::1] k1, k2, k3, k4, pv1, pv2, pv3, pv4
    pv1 = pv2 = pv3 = pv4 = np.empty(6)
    k1 = eval_k(pv0, E, B)
    for i in range(6): pv1[i] = pv0[i] + k1[i]*tstep/2.
    k2 = eval_k(pv1, E, B)
    for i in range(6): pv2[i] = pv0[i] + k2[i]*tstep/2.
    k3 = eval_k(pv2, E, B)
    for i in range(6): pv3[i] = pv0[i] + k3[i]*tstep
    k4 = eval_k(pv3, E, B)
    for i in range(6): pv4[i]  = pv0[i] + (k1[i]+2*(k2[i]+k3[i])+k4[i])*tstep/6.
    return pv4

cdef double[::1] new_velocity(double[::1] v1, int coll):
    cdef:
        double ke1, ke2, vmag1, vmag2
        double theta, phi, r, chi, psi
        double[::1] v2 = v1, energy_loss = np.array([1.,12.,16.])

    ke1  = kinetic(v1[0],v1[1],v1[2])
    ke2  = ke1 - energy_loss[coll]
    vmag1 = norm(v1[0],v1[1],v1[2])
    vmag2 = vmag1*sqrt(ke2/ke1)
    
    theta = acos(v1[0]/vmag1)
    phi   = atan(v1[1]/v1[0])
    r     = np.random.rand()
    chi   = acos(1-2*r/(1+8*ke1*(1-r)/27.21))
    psi   = 2*PI*np.random.rand()

    v2[0] = (cos(theta)*cos(phi)*sin(chi)*cos(psi) \
            -sin(phi)*sin(chi)*sin(psi) + sin(theta)*cos(phi)*cos(chi))*vmag2
    v2[1] = (cos(theta)*sin(phi)*sin(chi)*cos(psi) \
            +cos(phi)*sin(chi)*sin(psi) + sin(theta)*sin(phi)*cos(chi))*vmag2
    v2[2] = (-sin(theta)*sin(chi)*cos(psi) + cos(theta)*cos(chi))*vmag2

    return v2

def trace_single(double[::1] p0, double[::1] v0, 
        ElectricField E, MagneticField B, double[:,::1] db, double ng, 
        double tstep, size_t tnum, bint is_trajectory):
    cdef:
        size_t i
        list p_list = [], v_list = [], c_list = [], t_list = []
        double[::1] p1, v1, p2, v2, pv1, pv2
        int coll

    p1, v1 = p0, v0
    for i in xrange(tnum-1):
        coll = -1
        pv1 = np.concatenate([p1,v1])
        pv2 = move_onestep(pv1,tstep,E,B)
        p2, v2 = pv2[:3], pv2[3:]
        
        if not in_range(p2[0],p2[1]): break
        ke = kinetic(v2[0],v2[1],v2[2])
        pe = potential(p2[0],p2[1],p2[2],E)
        dx = distance(p1,p2)
        
        if np.random.rand() < coll_prob(ke,dx,db,ng):
            coll = coll_type(ke,db)
            v2 = new_velocity(v2,coll)
            if not is_trajectory:
                p_list.append(p2)
                v_list.append(v2)
                c_list.append(coll)
                t_list.append(tstep*(i+1))

        if is_trajectory:
            p_list.append(p2)
            v_list.append(v2)
            c_list.append(coll)
            t_list.append(tstep*(i+1))

        if pe+ke <26: break
        p1, v1 = p2, v2

    return t_list, p_list, v_list, c_list, i
    
cdef inline double kinetic(double vx, double vy, double vz):
    return 0.5*E_MASS*(vx**2 + vy**2 + vz**2)/E_CHARGE
    
cdef inline double potential(double x, double y, double z, ElectricField E):
    return -0.5*E.vol*(z-E.sh)**2/E.sh**2 if z<E.sh else 0.0

cdef inline double distance(double[::1] p1, double[::1] p2):
    return sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)

cdef double coll_prob(double ke, double dx, double[:,::1] db, double ng):
    cdef double sig_tt = np.interp(ke,db[:,0],db[:,4])
    return 1-exp(-ng*sig_tt*1.0E-20*dx)
    
cdef int coll_type(double ke, double[:,::1] db):
    cdef double sig_el, sig_ex, sig_tt, r
    sig_el = np.interp(ke,db[:,0],db[:,1])
    sig_ex = np.interp(ke,db[:,0],db[:,2])
    sig_tt = np.interp(ke,db[:,0],db[:,4])
    r = np.random.rand()
        
    if r>(sig_el+sig_ex)/sig_tt and ke>16: return 2
    if r>(sig_el)/sig_tt and ke>12:        return 1
    if ke>1:                               return 0
    
cdef bint in_range(double x, double y):
    return True if -0.1<x<0.1 and -0.03<y<0.03 else False
        
##########################################################################
def get_map_target(double[:,:] result, list selection, double resolution,
    double xmin, double xmax, double ymin, double ymax):
    cdef:
        double[:] xx, yy, cc
        double dx, dy
        int nx, ny, ti, tj
        size_t k
        double[:,::1] tmap
    
    xx, yy, cc = result[:,2], result[:,3], result[:,8]
    dx = dy = resolution
    nx, ny = int((xmax-xmin)/dx), int((ymax-ymin)/dy)
    tmap = np.zeros((ny,nx))

    for k in xrange(cc.shape[0]):
        if ["el","ex","iz"][int(cc[k])] in selection:
            ti = int((ymax-yy[k])/dy)
            tj = int((xx[k]-xmin)/dx)
            if 0<=ti<ny and 0<=tj<nx:
                tmap[ti,tj] += 1

    return np.asarray(tmap)

def get_map_substrate(double[:,::1] tmap, double gap, double resolution, 
    double txmin, double txmax, double tymin, double tymax,
    double sxmin, double sxmax, double symin, double symax):
    cdef:
        double tdx, tdy, sdx, sdy
        int tnx, tny, snx, sny, ti, tj, si, sj
        int t, s
        double[:,::1] smap
    
    tdx = tdy = resolution
    tnx, tny = int((txmax-txmin)/tdx), int((tymax-tymin)/tdy)
    sdx = sdy = resolution
    snx, sny = int((sxmax-sxmin)/sdx), int((symax-symin)/sdy)
    smap = np.zeros((sny,snx))

    for t in xrange(tnx*tny):
        ti = t//tnx
        tj = t%tnx 
        tx = txmin + tdx*(tj+0.5)
        ty = tymax - tdy*(ti+0.5)
        if tmap[ti,tj] > 0:
            for s in xrange(snx*sny):
                si = s//snx
                sj = s%snx
                sx = sxmin + sdx*(sj+0.5)
                sy = symax - sdy*(si+0.5)
                smap[si,sj] += tmap[ti,tj]*vf(tx,ty,sx,sy,gap)*tdx*tdy/sdx/sdy
    
    return np.asarray(smap)

cdef inline double vf(double x1, double y1, double x2, double y2, double d):
    return d**2/((x2-x1)**2 + (y2-y1)**2 + d**2)**2/PI
