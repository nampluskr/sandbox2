# cython: boundscheck = False
# cython: wraparound  = False
# cython: cdivision = True
import numpy as np
cimport numpy as np
from libc.math cimport log, sqrt, atan

########## Global constants for electrons:
DEF E_CHARGE = 1.602E-19 # [C]
DEF E_MASS   = 9.109E-31 # [Kg]

########## Electric Field [V/m]
cdef class ElectricField:
    cdef double vol, sh, tm
    def __cinit__(self, double voltage, double sheath, double target):
        self.vol = voltage
        self.sh = sheath
        self.tm = target
    cpdef double Ex(self, double x, double y, double z):
        return 0.0
    cpdef double Ey(self, double x, double y, double z):
        return 0.0
    cpdef double Ez(self, double x, double y, double z):
        if z > self.tm+self.sh:
            return 0.0
        else:
            return 2.0*self.vol*(self.sh+self.tm-z)/self.sh**2

########## Magnetic field [T] (Analytic expression)
cdef class MagneticField:
    cdef object mags
    def __cinit__(self, *magnets):
        self.mags = magnets
    cpdef double Bx(self, double x, double y, double z):
        cdef double sum_bx = 0.0
        cdef Magnet m
        for m in self.mags:
            sum_bx += m.Bx(x,y,z)
        return sum_bx
    cpdef double By(self, double x, double y, double z):
        cdef double sum_by = 0.0
        cdef Magnet m
        for m in self.mags:
            sum_by += m.By(x,y,z)
        return sum_by
    cpdef double Bz(self, double x, double y, double z):
        cdef double sum_bz = 0.0
        cdef Magnet m
        for m in self.mags:
            sum_bz += m.Bz(x,y,z)
        return sum_bz

cdef class Magnet:
    cdef double x0, y0, z0, a, b, c, scale
    def __cinit__(self, x0, y0, z0, length, width, height, scale):
        self.x0,self.y0,self.z0 = x0, y0, z0
        self.a, self.b, self.c  = length, width, height
        self.scale = scale
    cdef double Bx(self, double _x, double _y, double _z):
        cdef double x = _x-self.x0, y = _y-self.y0, z = _z-self.z0, bx
        bx = f1(self.a-x,y,z,self.c) + f1(self.a-x,self.b-y,z,self.c) \
           - f1(x,y,z,self.c)        - f1(x,self.b-y,z,self.c) \
           - f1(self.a-x,y,z,0)      - f1(self.a-x,self.b-y,z,0) \
           + f1(x,y,z,0)             + f1(x,self.b-y,z,0)
        return -self.scale*bx*0.5
    cdef double By(self, double _x, double _y, double _z):
        cdef double x = _x-self.x0, y = _y-self.y0, z = _z-self.z0, by
        by = f1(self.b-y,x,z,self.c) + f1(self.b-y,self.a-x,z,self.c) \
           - f1(y,x,z,self.c)        - f1(y,self.a-x,z,self.c) \
           - f1(self.b-y,x,z,0)      - f1(self.b-y,self.a-x,z,0) \
           + f1(y,x,z,0)             + f1(y,self.a-x,z,0)
        return -self.scale*by*0.5
    cdef double Bz(self, double _x, double _y, double _z):
        cdef double x = _x-self.x0, y = _y-self.y0, z = _z-self.z0, bz
        bz = f2(y,self.a-x,z,self.c) + f2(self.b-y,self.a-x,z,self.c) \
           + f2(x,self.b-y,z,self.c) + f2(self.a-x,self.b-y,z,self.c) \
           + f2(self.b-y,x,z,self.c) + f2(y,x,z,self.c) \
           + f2(self.a-x,y,z,self.c) + f2(x,y,z,self.c) \
           - f2(y,self.a-x,z,0)      - f2(self.b-y,self.a-x,z,0) \
           - f2(x,self.b-y,z,0)      - f2(self.a-x,self.b-y,z,0) \
           - f2(self.b-y,x,z,0)      - f2(y,x,z,0) \
           - f2(self.a-x,y,z,0)      - f2(x,y,z,0)
        return -self.scale*bz

cdef inline double norm(double x, double y, double z):
    return sqrt(x*x + y*y + z*z)

cdef inline double f1(double x, double y, double z, double h):
    return log((norm(x,y,z-h)-y)/(norm(x,y,z-h)+y))

cdef inline double f2(double x, double y, double z, double h):
    return atan((x/y)*(z-h)/norm(x,y,z-h))

##########################################################################
cdef eval_k(double[::1] u, ElectricField E, MagneticField B):
    cdef double[::1] k
    k = np.empty(len(u))
    k[0] = u[3]
    k[1] = u[4]
    k[2] = u[5]
    k[3] = -(E.Ex(u[0],u[1],u[2]) + u[4]*B.Bz(u[0],u[1],u[2]) \
         - u[5]*B.By(u[0],u[1],u[2]))*(E_CHARGE/E_MASS)
    k[4] = -(E.Ey(u[0],u[1],u[2]) + u[5]*B.Bx(u[0],u[1],u[2]) \
         - u[3]*B.Bz(u[0],u[1],u[2]))*(E_CHARGE/E_MASS)
    k[5] = -(E.Ez(u[0],u[1],u[2]) + u[3]*B.By(u[0],u[1],u[2]) \
         - u[4]*B.Bx(u[0],u[1],u[2]))*(E_CHARGE/E_MASS)
    return np.asarray(k)

cdef void solve_rk4(double[::1] t, double[::1] u0, ElectricField E, MagneticField B, \
    size_t NDATA, size_t NEQN, double[:,::1] uu):
    cdef:
        size_t i, j
        double h
        double[::1] u1, u2, u3, u4
        double[::1] k1, k2, k3, k4
    u1 = u2 = u3 = u4 = np.empty(NEQN)
    uu[0] = u0
    for i in range(NDATA-1):
        h = t[i+1]-t[i]
        for j in range(NEQN):
            u1[j] = uu[i,j]
        k1 = eval_k(u1, E, B)
        for j in range(NEQN):
            u2[j] = uu[i,j] + k1[j]*h/2.
        k2 = eval_k(u2, E, B)
        for j in range(NEQN):
            u3[j] = uu[i,j] + k2[j]*h/2.
        k3 = eval_k(u3, E, B)
        for j in range(NEQN):
            u4[j] = uu[i,j] + k3[j]*h
        k4 = eval_k(u4, E, B)
        for j in range(NEQN):
            uu[i+1,j] = uu[i,j] + (k1[j]+2*(k2[j]+k3[j])+k4[j])*h/6.

cdef class ElectronTrace:
    cdef:
        size_t NDATA, NEQN
        double[::1] t, u0
        double[:,::1] uu
    def __cinit__(self, t, u0):
        self.NDATA = len(t)
        self.NEQN  = len(u0)
        self.t = t
        self.u0 = u0
        self.uu = np.empty((self.NDATA,self.NEQN))
    cpdef solve(self, ElectricField E, MagneticField B):
        solve_rk4(self.t, self.u0, E, B, self.NDATA, self.NEQN, self.uu)
        return np.asarray(self.uu)
