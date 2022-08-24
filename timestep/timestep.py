from numpy import linspace,exp
import matplotlib.pyplot as plt

step = 1.0
scale = 0.1


dseps = linspace(0.0,scale,1000)


def timestep(dsep,scale):
  return (dsep / scale) * 10**(dsep)

plt.plot(dseps,timestep(dseps,scale))

plt.savefig("test.pdf")