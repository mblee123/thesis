# All units should be in in m, s, and kg
# Code created by Alec Griffith, contact: alecrbgriffith@gmail.com if help
#n necessary
import numpy as np
from rayTrace import *
from math import ceil
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

traceTable = None
space = None

c = 3*10**8
kB = 1.380648e-23

class Particle:
    
    position = np.array([0,0,0])
    velocity = np.array([0,0,0])
    mass = 0
    radius = 0
    shell = Shell(position, radius, 1.0, 1.0)
    
    def __init__(self, p, v, m, r, no, ni):

        self.position = p
        self.velocity = v
        self.mass = m
        self.radius = r
        self.shell = Shell(p, r, no, ni)
        
class Environment:
    
    
    gravity = 9.8
    density = 0.0
    pressure = 0.0
    temperature = 0.0
    viscosity = 0.0

    def __init__(self, d, p, t, nu):
        
        self.density = d
        self.pressure = p
        self.temperature = t
        self.viscosity = nu
        self.gravity = 9.81


def produceTraceTable(sphere, nRays):
    global traceTable
    
    #Use a basic unit sphere for trace table
    traceSphere = Shell(np.array([0,0,0]), 1.0, sphere.no, sphere.ni)
    traceTable = np.zeros((nRays,4))
    
    # go up to paralle from both directions
    maxTheta = np.pi/2.0
    traceTable[:,0] = np.linspace(-1.0*maxTheta, maxTheta, nRays)
    
    for i,theta in enumerate(traceTable[:,0]):
        
        direction = np.array([-np.sin(theta),0,np.cos(theta)])
  
        origin =  - direction
        
        r = Ray(origin, direction, 1.0, False)
        traceTable[i,1:] = propogate(traceSphere, r, 1e-5)

#create our points based on sphere and coordinates desired
def produceIntegrationSpace(sphere, nCoord):
    global space
    
    thetas = np.linspace(0, np.pi, nCoord+1)

    #infintesmial angle chunk
    dT = np.pi/float(nCoord)
    
    #get rid of one of the pole thetas and shif
    thetas = thetas[:-1]+dT/2

    #get prepared to append
    space = []
    
    for theta in thetas:
        
        #average from theta-dT/2 to theta+dT/2 for better precision
        sinT = 2*np.sin(theta)*np.sin(dT/2)/dT
        
        #approximately grab the same area with each point, ie same dl going
        #around each semicircle
        circumference = 2*np.pi*np.sin(theta)
        nPhis = max(2,ceil(circumference/dT))
        dP = 2*np.pi/float(nPhis)
        phis = np.linspace(0, 2*np.pi, nPhis+1)
        phis = phis[:-1] 

        for phi in phis:
            
            space.append([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta),dT*dP*sinT])
    
    #adjust for area by multiplying out by sphere radius,
    # the first three columns get multiplied only once as they are positions
    # the last gets multiplied an extra time because it is the area        
    space = np.array(space)
    space *= sphere.radius
    space[:,3] *= sphere.radius


    
def laserForce(particle, beam):
    
    normals = -space[:,:-1]/particle.radius
    height,width = space.shape
    values = np.zeros((height,4))

    #vectorized to make faster
    values = beam.optIntensity(space[:,:-1]+particle.position)
  
    #take the dotproduce of various vectors
    dotProduct = np.einsum('ij,ij->i',normals,values[:,:-1])
    #Theta calculation to pick up sign
    crossProduct = np.cross(normals,values[:,:-1])
    crossMagnitudes = np.linalg.norm(crossProduct,axis=1)
    thetas = np.arcsin(crossMagnitudes)
    
    #zeroOut matrix which we will use to not grab the out going rays
    zeroOut = 0 <= dotProduct
    zeroOut = np.array([zeroOut]).T

    #net changes
    #note some of these vectors are going out and wrongly interpolated, we will
    #set them to zero later carefully
    xs = np.interp(thetas,traceTable[:,0],traceTable[:,1])
    ys = np.interp(thetas,traceTable[:,0],traceTable[:,2])
    zs = np.interp(thetas,traceTable[:,0],traceTable[:,3])
    interpRayTrace = np.array([xs,ys,zs])
    interpRayTrace = interpRayTrace.T
    
    #rotate into proper frame
    #find the vector each theta corrsponds to then figure out the rotation we must get to transform it to the thing we want
    #http://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    angleBeams = np.array([crossMagnitudes, np.zeros(thetas.shape), dotProduct])
    angleBeams = angleBeams.T
    vs = np.cross(angleBeams, values[:,:-1])
    cs = np.einsum('ij,ij->i',angleBeams,values[:,:-1])
    skews = np.array([[np.zeros(vs[:,0].shape),-vs[:,2],vs[:,1]],
                      [vs[:,2],np.zeros(vs[:,0].shape),-vs[:,0]],
                      [-vs[:,1],vs[:,0],np.zeros(vs[:,0].shape)]])
    #einsum does the dot product of each matrix
    ids = np.zeros(skews.shape)
    ids[[0,1,2],[0,1,2],:] = 1
    skews2 = np.einsum('ijl,jkl->ikl',skews,skews)
    rs = ids + skews + skews2/(1+cs)

    # the ith matrix is Rs[:,:,i], performs a rotation back to original angle
    outResults = np.einsum('ijl,lj->li',rs,interpRayTrace)
    
    # now transform to right position on the sphere.
    zs = np.zeros(ids.shape)
    zs[2,2,:] = 1
    zs[[0,1],[0,1],:] = -normals[:,0]
    zs[0,1,:] = normals[:,1]
    zs[1,0,:] = -normals[:,1]

    #Secondary rotation
    outResults = np.einsum('ijl,lj->li',zs,interpRayTrace)
    
    netResults = values[:,:-1]-outResults
    #get rid of outgoing directionOnes
    netResults *= zeroOut
    #multiply by the appropriate beam intensities
    netResults *= np.array([values[:,-1]]).T
    #multiply by the appropriate areas
    netResults *= np.array([space[:,-1]]).T
    #collapse them down to one result, now in terms of power
    netResults = np.sum(netResults,axis=0)
    #convert from energy per second to momentum per second
    netResults /= c
    
    return netResults
    
def drag(particle, environment):
    
    #Stoke's law drag
    baseForce = -6.0*np.pi*environment.viscosity*particle.radius*particle.velocity

    fluidParticleRadius = 1.785e-10
    #mean free path
    mfp = kB*environment.temperature/(4*np.sqrt(2)*np.pi*fluidParticleRadius**2*environment.pressure)
    #Knusden number
    k = mfp/particle.radius
    
    #Cunningham correction
    C = 1 + k*(1.257+0.4*np.exp(-1.1*k))
    
    #implement the correction
    return baseForce/C

def gravity(particle,environment):
    
    #gravity alignment, can be changed if one wants to rotate the setup
    return particle.mass*environment.gravity*np.array([0,0,-1])
    
def accel(particle,beam,environment):
    
    #add all forces
    return (gravity(particle,environment)+drag(particle,environment)+laserForce(particle,beam))/particle.mass
    
def timeStep(particle,beam,environment, step):
    
    #basic implicit euler
    particle.velocity += accel(particle,beam,environment)*step
    particle.position += particle.velocity*step
    return particle
    
def run(particle, beam, environment, time, step):
    
    #step a bunch of times
    times = np.arange(0,time,step)
    positions = np.zeros((times.shape[0],3))
    
    for i in range(times.shape[0]):
        positions[i,:] = particle.position
        particle = timeStep(particle,beam,environment,step)
        
    return times,positions
    
def main():
    
    #really just a basic rundown of doing some test versions, should be a good
    #idea of how to run the program, can also look into test ray trace
    origin = np.array([0.0,0.0,0.0])
    o = np.array([5e-6,5e-6,1000e-6])
    ov = np.array([0.0,0.0,0.0])
    e = Environment( 1.17e-3, 100, 300, 1.99e-5)
    
    radius = 5e-6
    mass = 1.5*1197.2 * 4.0/3.0 * np.pi * radius**3
    p = Particle(o, ov, mass, radius , 1.0, 1.38)
    produceIntegrationSpace(p.shell, 30)
    produceTraceTable(p.shell,200)
    beam = Gaussian(1e-1, origin, 532e-9, 3e-6) 
    
    times, positions = run(p,beam,e,5e-1,1e-3)
    plt.plot(times,positions[:,0])
    plt.savefig("xE.png",dpi=300)
    plt.clf()

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('x position')
    ax.set_ylabel('y position')
    ax.set_zlabel('z position')
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(700,800)
    ax.scatter(positions[:,0]*1e6, positions[:,1]*1e6,positions[:,2]*1e6)
    fig.savefig("m_1_5_t_5.png")
    plt.clf()


    plt.plot(times,positions[:,1])
    plt.savefig("yE.png",dpi=300)
    plt.clf()
    radii = np.sqrt(positions[:,0]**2+positions[:,1]**2)
    plt.plot(times,radii)
    plt.savefig("rE.png",dpi=300)
    plt.clf()
    plt.plot(times,positions[:,2])
    plt.savefig("zE.png",dpi=300)
    plt.clf()
    plt.xlabel("radial position (micro-meters)")
    plt.ylabel("vertical position (micro-meters)")
    plt.plot(radii*1e6,positions[:,2]*1e6)
    plt.savefig("trajR.png",dpi=300)
    plt.clf()
    plt.xlabel("x position (micro-meters)")
    plt.ylabel("z position (micro-meters)")
    plt.plot(positions[:,0]*1e6,positions[:,2]*1e6)
    plt.savefig("trajX.png",dpi=300)
    plt.clf()
    
    
if __name__ == "__main__":
    main()