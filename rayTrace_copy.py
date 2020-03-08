import numpy as np
from abc import ABCMeta,abstractmethod

class Ray:

    #origin of ray
    origin = np.array([0,0,0])
    #unit vector pointing direction of ray
    direction = np.array([0,0,1])
    #intensity variable
    magnitude = 1.0
    #inside sphere or not
    inside = False

    def __init__(self, o, d, m, i):

        self.origin = o
        self.direction = d
        self.magnitude = m
        self.inside = i

class Shell:

    #origin of spherical shell
    origin = np.array([0,0,0])
    #radius of shell
    radius = 1.0
    #index of refraction outside
    no = 1.0
    #index of refraction inside
    ni = 1.0

    def __init__(self, c, r, nOutside, nInside):

        self.origin = c
        self.radius = r
        self.no = nOutside
        self.ni = nInside

# abstract class for which we promise to implement the following 
class Beam:
    __metaclass__ = ABCMeta
    
    #Power of laser beam
    power = None
    #Base point for laaser beam
    origin = None
    #wavelength
    l = None
    
    #Return ray with magnitude, direction, location like infinite plane wave
    @abstractmethod
    def intensity(self, position):
        pass
    
    #Return a 2d matrix of outgoing directions and intensities, vectorized for efficiency
    @abstractmethod
    def optIntensity(self, position):
        pass

class Gaussian(Beam):
    
    #inherited variables
    power = 0
    origin = 0
    wavelength = 0
    
    #focus beam radius
    waist0 = 0
    #focus position
    z0 = 0
    
    def __init__(self, p, o, l, w0):
        
        self.power = p
        self.origin = o
        self.wavelength = l
        self.waist0 = w0
        self.z0 = np.pi*w0**2/l
    
    def intensity(self, position):
        
        #shift the coordinate system
        position = position-self.origin
        
        #find parameters radius squared, radius, beam waist
        rho2 =  position[0]**2+position[1]**2
        w =self.waist(position)
        
        #get intensity
        I = 2*self.power/(np.pi*w**2)*np.exp(-2*rho2/w**2)
        
        #get direction
        sinT = position[1]
        cosT = position[0]
        
        differentialOut = position[2]/(self.z0**2+position[2]**2)
        
        direction = np.array([differentialOut*cosT,differentialOut*sinT,1])/np.sqrt(rho2*differentialOut**2+1)
        
        #assume the ray is outside the sphere before further notice
        return Ray(position+self.origin, direction, I, False)
        
    def waist(self, relPosition):
        
        return self.waist0*np.sqrt(1+(relPosition[2]/self.z0)**2)
        
    #vectorized and optimized intensity
    def optIntensity(self, position):
        
        #shift the coordinate system
        position = position-self.origin
        
        #find parameters radius squared, radius, beam waist
        rho2 =  position[:,0]**2+position[:,1]**2
        rho = np.sqrt(rho2)
        w = self.waist0*np.sqrt(1+(position[:,2]/self.z0)**2)
        
        #get intensity
        I = 2*self.power/(np.pi*w**2)*np.exp(-2*rho2/w**2)
        
        #get direction
        sinT = position[:,1]/rho
        cosT = position[:,0]/rho
        
        differentialOut = position[:,2]*rho/(self.z0**2+position[:,2]**2)
        normalizer = np.sqrt(differentialOut**2+1)
        
        direction = np.array([differentialOut*cosT/normalizer,differentialOut*sinT/normalizer,1/normalizer])
        results = np.array([direction[0], direction [1],direction[2], I])
        

        return results.T
        

def collision(sphere, beam):
    
    #find the position, then find possible coollision roots
    diff = beam.origin - sphere.origin
    b = np.dot(beam.direction, diff)
    
    root = np.sqrt(b**2 - np.dot(diff,diff) + sphere.radius**2)

    d1 = -b + root
    d2 = -b - root
    
    #find the first real collision, not including ultra short length ones
    #most likely from mispositioning
    tuner = 1e-7
    
    if d1 > sphere.radius*tuner and (d1 < d2 or d2 <= sphere.radius*tuner):

        return d1

    elif d2 > sphere.radius*tuner:

        return d2
        
    else:
        
        return None

def reflectedRay(sphere, beam, position, normal):

    #find the reflected ray location and direction
    reflectedDirection = beam.direction - 2 * np.dot(beam.direction, normal) * normal
    reflectedDirection /= np.linalg.norm(reflectedDirection)
    reflectedOrigin = position

    return Ray(reflectedOrigin, reflectedDirection, beam.magnitude, beam.inside)

def refractedRay(sphere, beam, position, normal):

    #incident angle cos
    cosIncident = np.abs(np.dot(beam.direction, normal))

    #default outisde to inside
    n1 = sphere.no
    n2 = sphere.ni
    
    #inside to outside
    if beam.inside:

        n1 = sphere.ni
        n2 = sphere.no

    # see bookmarked stanford assignment
    nr = n1 / n2
    refractedDirection = nr * beam.direction + (nr * cosIncident - np.sqrt(1 - nr**2 * (1 - cosIncident))) * normal
    refractedDirection /= np.linalg.norm(refractedDirection)

    refractedOrigin = position

    return Ray(refractedOrigin, refractedDirection, beam.magnitude, not beam.inside)

#This assumes unpolarized light
def reflectionCoefficients(sphere, beam, position, normal):

    direction = -beam.direction

    cosI = np.abs(np.dot(direction, normal))
    sinI = np.linalg.norm(np.cross(direction, normal))    

    #default outisde to inside
    n1 = sphere.no
    n2 = sphere.ni

    #inside to outside
    if beam.inside:

        n1 = sphere.ni
        n2 = sphere.no
    
    nr = n1  / n2

    #determine if all reflected
    criticalParam = (nr * sinI)**2
    if criticalParam >= 1:
        
        R = 1

    else:
        
        cosT = np.sqrt(1 - criticalParam**2)

        Rs = np.abs((n1 * cosI - n2 * cosT) / (n1 * cosI + n2 * cosT))**2
        Rp = np.abs((n1 * cosT - n2 * cosI) / (n1 * cosT + n2 * cosI))**2

        R = (Rs + Rp) / 2

    R=0.96

    return R, 1 - R




def split(sphere, beam):
    
    #find distance to the sphere
    
    d = collision(sphere, beam)


    if d is None:

            return None, None

    else:

        #find the intersection
        position = beam.origin + d * beam.direction
        normal = (position - sphere.origin) / sphere.radius
        
        #inside to outside flip sign of normal
        if beam.inside:

            normal *= -1

        refr = refractedRay(sphere, beam, position, normal)
        refl = reflectedRay(sphere, beam, position, normal)

        # next alter the intensities as appropriate
        R,T = reflectionCoefficients(sphere, beam, position, normal)

        refr.magnitude *= T
        refl.magnitude *= R
        

        return refr,refl

# variables for plotting, currently saved, later remove
# these plotting variables are commented out as they are incredibly innefficient
# if implemented correctly you can create branching diagrams, but most
# of the plotting information was removed, so I wouldn't re-activate unless
# absolutely necessary
"""
origins = []
ends = []
opacities = []
"""

def propogate(sphere, beam, criticalMagnitude):
    global origins, ends, opacities
    
    # get relevant beams, prepare accumuolating variable
    refr, refl = split(sphere, beam)
    weightedDirection = np.zeros((3))
    
    """
    dist = 1000
    """
    
    # if we don't collide, then add the beam
    # else propogate the post collision rays if they have adequate magnitude
    if refr is None and refl is None:
        
        weightedDirection = beam.magnitude * beam.direction
    
    else:
        
        """     
        seperation = beam.origin - refr.origin
        dist = np.sqrt(np.dot(seperation,seperation))
        """
        # if we still care about the reflection, propogate it, similarly refraction
        if refr.magnitude > criticalMagnitude: 

            weightedDirection += propogate(sphere, refr, criticalMagnitude)
        
        else:

            weightedDirection += refr.magnitude * refr.direction
            
        if refl.magnitude > criticalMagnitude:

            weightedDirection += propogate(sphere, refl, criticalMagnitude)

        else:
            
            weightedDirection += refl.magnitude * refl.direction

    """
    origins.append(beam.origin)
    ends.append(beam.direction * dist + beam.origin)
    opacities.append(beam.magnitude)
    """
    
    return weightedDirection

