import matplotlib.pyplot as plt
from timeit import default_timer as timer

from rayTrace import *
from trap import *


# A bunch of little tests I made which helped in making plots as documented in
# the thesis, I haven't commented these as they aren't fundamental, but
# this should serve as a good example program of creating plots with different
# parameters if you need it

def testGaussian():
    origin = np.array([0, 0, 0])

    for power in range(4):
        g = Gaussian(1, origin, 1, 2 * 1.5 ** (-1 * power))

        maxPoint = 3.
        resolution = 20.

        points = np.arange(-1. * maxPoint * (1.0 + 2.0 / resolution), maxPoint * (1.0 + 2.0 / resolution),
                           2 * maxPoint / resolution)
        x, z = np.meshgrid(points, points)
        valX, valZ = np.meshgrid(points, points)
        mag, z = np.meshgrid(points, points)

        xLen, zLen = x.shape
        for i in range(xLen):
            for j in range(zLen):
                p = np.array([x[i, j], 0, z[i, j]])
                r = g.intensity(p)
                mag[i, j] = r.magnitude
                valX[i, j] = r.direction[0] * r.magnitude
                valZ[i, j] = r.direction[2] * r.magnitude
        plt.contourf(x, z, mag)
        plt.quiver(x, z, valX, valZ)
        plt.axis([-maxPoint, maxPoint, -maxPoint, maxPoint])
        plt.axis("off")
        plt.savefig("testGaussian" + str(power) + ".png", dpi=300, bbox_inches="tight")
        plt.clf()


def testRegularParams():
    origin = np.array([0.0, 0.0, 0.0])

    g = Gaussian(1e-1, origin, 532e-9, 3e-6)

    maxPoint = 25e-6
    resolution = 20.

    points = np.arange(-1. * maxPoint * (1.0 + 2.0 / resolution), maxPoint * (1.0 + 2.0 / resolution),
                       2 * maxPoint / resolution)
    x, z = np.meshgrid(points, points)
    valX, valZ = np.meshgrid(points, points)
    mag, z = np.meshgrid(points, points)

    xLen, zLen = x.shape
    for i in range(xLen):
        for j in range(zLen):
            p = np.array([x[i, j], 0, z[i, j]])
            r = g.intensity(p)
            mag[i, j] = r.magnitude
            valX[i, j] = r.direction[0] * r.magnitude
            valZ[i, j] = r.direction[2] * r.magnitude
    x *= 1e6
    z *= 1e6
    plt.contourf(x, z, mag)
    plt.quiver(x, z, valX, valZ)
    plt.axis([-maxPoint * 1e6, maxPoint * 1e6, -maxPoint * 1e6, maxPoint * 1e6])
    particle = plt.Circle((0, 12.5), radius=5, color='white', fill=True, alpha=0.75)
    plt.axes().set_aspect('equal')
    plt.gca().add_patch(particle)
    plt.xlabel("x position (micrometers)")
    plt.ylabel("z position (micrometers)")
    plt.savefig("gaussianActual.png", dpi=300, bbox_inches="tight")
    plt.clf()


def testCutoff():
    s1 = Shell(np.array([0.0, 0.0, 3.0]), 1.0, 1.0, 2.0)

    z = 2.0

    trials = 200
    maxTheta = np.pi / 2
    thetas = np.linspace(0., maxTheta, trials)
    results = np.zeros((3, trials))
    for power in range(4):
        cutoff = 10 ** (-power - 1)

        start = timer()
        for i in range(trials):
            x = z * (np.tan(maxTheta * float(i) / float(trials)))

            origin = np.array([x, 0, 0])
            direction = np.array([-x, 0, z])
            direction = direction / np.linalg.norm(direction)
            r1 = Ray(origin, direction, 1, False)

            # refr, refl = split(s1,r1)
            # results[0,i] = refr.magnitude
            # results[1,i] = refl.magnitude
            results[:, i] = propogate(s1, r1, cutoff)

        trace_time = timer() - start
        # print("Ray tracing " + str(trials) + " rays with a " + str(cutoff) + " cutoff took in total " + str(trace_time))

        plt.plot(thetas, results[0, :], label='X')
        plt.plot(thetas, results[1, :], label='Y')
        plt.plot(thetas, results[2, :], label='Z')
        plt.axis([0, maxTheta, -1, 1])
        plt.ylabel("outgoing magnitude")
        plt.xlabel("angle of incidence (radians)")
        plt.title("outgoing magnitude vs. angle at cutoff " + str(cutoff))
        plt.legend()
        plt.savefig("testCutoff" + str(power + 1) + ".png", dpi=300)
        plt.clf()

    """
    x = z*(np.tan(1.11066407))
    origin = np.array([x,0,0])
    direction = np.array([-x,0,z])
    direction = direction/np.linalg.norm(direction)
    r1 = Ray(origin,direction,1)
    
    temp = propogate(s1, r1, cutoff)
    print temp[0]
    
    
    for i in range(len(opacities)):
        xs = [origins[i][0],ends[i][0]]
        zs = [origins[i][2],ends[i][2]]
        opacity = opacities[i]
        plt.plot(xs,zs,color='#00FF00',linewidth=5.0*opacity,alpha=1)
    
    circ=plt.Circle((0,3),radius=1,color='black',linewidth=4.0,fill=False)
    plt.gca().add_patch(circ)
    plt.axis([-3, 3, 0, 6])
    fig = plt.gcf()
    fig.set_size_inches(12, 12)
    plt.axis('off')
    plt.savefig("second.png",dpi=100)
    """


def testLaserForce():
    maxI = 121
    results = np.zeros((maxI, 3))
    xaxis = np.array(range(maxI)) + 1

    origin = np.array([0.0, 0.0, 0.0])
    o = np.array([25e-6, 0.0, 1000e-6])
    ov = np.array([0.0, 0.0, 0.0])
    radius = 5e-6
    mass = 1197.2 * 4.0 / 3.0 * np.pi * radius ** 3
    p = Particle(o, ov, mass, radius, 1.0, 1.38)
    produceIntegrationSpace(p.shell, 30)
    produceTraceTable(p.shell, 200)
    beam = Gaussian(1e-1, origin, 532e-9, 3e-6)

    for i in range(maxI):
        produceIntegrationSpace(p.shell, i + 1)
        start = timer()
        results[i, :] = laserForce(p, beam)
        force_time = timer() - start
        # print("Integration with resolution " + str(i+1) + " took in total " + str(force_time))

    plt.plot(xaxis, results[:, 0], label='X')
    plt.plot(xaxis, results[:, 1], label='Y')
    plt.plot(xaxis, results[:, 2], label='Z')
    plt.ylabel("laser force (newtons)")
    plt.xlabel("n")
    plt.axis([1, 121, -0.85e-11, 0.85e-11])
    # plt.title("integration convergence")
    plt.legend()
    plt.savefig("integrationConvergence.png", dpi=300)
    plt.clf()


def testConvergence():
    plt.clf()
    testI = 1200

    maxI = 300
    results = np.zeros((maxI, 3))
    xaxis = np.array(range(maxI)) + 1

    origin = np.array([0.0, 0.0, 0.0])
    o = np.array([25e-6, 0.0, 1000e-6])
    ov = np.array([0.0, 0.0, 0.0])
    radius = 5e-6
    mass = 1197.2 * 4.0 / 3.0 * np.pi * radius ** 3
    p = Particle(o, ov, mass, radius, 1.0, 1.38)
    produceIntegrationSpace(p.shell, 30)
    produceTraceTable(p.shell, 200)
    beam = Gaussian(1e-1, origin, 532e-9, 3e-6)

    start = timer()
    produceIntegrationSpace(p.shell, testI)
    resultsPrimary = laserForce(p, beam)
    force_time = timer() - start
    print("Integration with resolution " + str(testI) + " took in total " + str(force_time))

    for i in range(maxI):
        produceIntegrationSpace(p.shell, i + 1)
        start = timer()
        results[i, :] = laserForce(p, beam)
        force_time = timer() - start
        # print("Integration with resolution " + str(i+1) + " took in total " + str(force_time))

    results -= resultsPrimary
    results = np.abs(results)
    results = np.log(results)
    xaxis = np.log(xaxis) * 2
    plt.plot(xaxis, results[:, 0], label='X')
    plt.plot(xaxis, results[:, 1], label='Y')
    plt.plot(xaxis, results[:, 2], label='Z')
    plt.ylabel("ln(E)")
    plt.xlabel("ln(n)")
    plt.axis([0, np.log(maxI + 1), -36, -24])
    # plt.title("integration convergence")
    plt.legend()
    plt.savefig("testConvergence.png", dpi=300)
    np.savetxt("ln(n).csv", xaxis)
    np.savetxt("ln(E).csv", results, delimiter=",")
    plt.clf()


def main():
    testGaussian()
    testCutoff()
    testLaserForce()
    testConvergence()
    testRegularParams()


if __name__ == "__main__":
    main()
