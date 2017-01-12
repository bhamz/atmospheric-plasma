import numpy as np
import scipy.integrate as integr8
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# radius, inclination, aziumuth
def spher_to_cart(r,theta,phi):
    x = r*np.sin(phi)*np.cos(theta)
    y = r*np.sin(phi)*np.sin(theta)
    z = r*np.cos(phi)
    return x,y,z

# velocity vector components from pitch angle and ini v
def traj(v,pitch):
    vx = 0
    vy = v*np.sin(pitch)*np.pi/180
    vz = v*np.cos(pitch)*np.pi/180
    return vx, vy, vz

# 2nd order diff eq for plasma motion
def plasmotion(y,t,b0,qmratio,rad):
    # unpack data
    x,y,z,vx,vy,vz = y
    b0,qmratio,rad = params
    # roots
    r = np.sqrt((x**2)+(y**2)+(z**2))
    factor = b0*(rad**3)/(r**5)
    # magnetic field
    bx = -3*x*z*factor
    by = -3*y*z*factor
    bz = ((x**2)+(y**2)-2*(z**2))*factor
    # accelerations
    ax = qmratio*(vy*bz - vz*by)
    ay = qmratio*(vz*bx - vx*bz)
    az = qmratio*(vx*by - vy*bx)
    # return derivatives
    return [vx,vy,vz,ax,ay,az]

# parameters
r_e = 6378137   # earth radius
q_elem = 1.602e-19 # elementary charge
B0 = 3.07e-5 # magnetic field strength @ equator
m_p = 1.672e-27 #proton mass
m = m_p
q_m = q_elem/m  #charge to mass ratio
energies_EV = 5e7; energies_j = energies_EV*q_m # converts EV -> Joules
c = 299792458 # speed of light


# intial conditions
v0 = c/np.sqrt(1+(m*c**2)/energies_j)   # energy -> velocity
pitch_angle = 35 # particle trajectory angle from field lines
rad_start = 4*r_e # initial particle dist from earth center
x0,y0,z0 = spher_to_cart(rad_start,0,np.pi/2) # particle start positions
vx0,vy0,vz0 = traj(v0,pitch_angle)  # gets initial directional velocities
ICs = [x0,y0,z0,vx0,vy0,vz0]    # initial condition vector
t_stop = 4000; dt = 0.05    # time vector inputs
t = np.linspace(0,t_stop,int(t_stop/dt)) #time vector

fig = plt.figure()
ax = fig.add_subplot(111,projection = '3d')

params = (B0,q_m,r_e)
dynamics = integr8.odeint(plasmotion,ICs,t,params)
ax.plot(dynamics[:,0]/r_e,dynamics[:,1]/r_e,dynamics[:,2]/r_e)
plt.show()
