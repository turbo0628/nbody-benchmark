import taichi as ti
import time

ti.init(arch=ti.cuda)

nBodies = 30000
softening = 1e-9
dt = 0.01

bodies = ti.field(shape=(nBodies, 2, 3), dtype=ti.float32)

@ti.kernel
def bodyForce():
    for i in range(nBodies):
        Fx = 0.0
        Fy = 0.0
        Fz = 0.0;
        for j in range(nBodies):
            vec = ti.Vector([0.0, 0.0, 0.0])
            dx = bodies[j, 0, 0] - bodies[i, 0, 0]
            dy = bodies[j, 0, 1] - bodies[i, 0, 1]
            dz = bodies[j, 0, 2] - bodies[i, 0, 2]
            distSqr = dx * dx + dy * dy + dz * dz + softening
            invDist = 1.0 / ti.sqrt(distSqr)
            invDist3 = invDist * invDist * invDist
            Fx += dx * invDist3
            Fy += dy * invDist3
            Fz += dz * invDist3
        bodies[i, 1, 0] += dt * Fx;
        bodies[i, 1, 1] += dt * Fy;
        bodies[i, 1, 2] += dt * Fz;

    for i in range(nBodies):
        bodies[i, 0, 0] += bodies[i, 1, 0] * dt;
        bodies[i, 0, 1] += bodies[i, 1, 1] * dt;
        bodies[i, 0, 2] += bodies[i, 1, 2] * dt;

@ti.kernel
def randomizeBodies():
    for i, j, k in bodies:
        bodies[i, j, k] = 2.0 * ti.random(dtype=ti.float32) - 1.0

@ti.kernel
def substep():
    body_force()

def run_nbody():
    nIters = 1024
    randomizeBodies()
    bodyForce() # warm-up
    st = time.time()
    for i in range(nIters):
        bodyForce()
    ti.sync()
    et = time.time()

    avg_time =  (et - st) * 1000.0 / nIters
    print("Finishing...time {}ms".format(avg_time))
    print("Nbody speed {} Billion bodies per second.".format(1e-6 * nBodies * nBodies / avg_time))
run_nbody()
