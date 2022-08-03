from amuse.units import units
from amuse.lab import new_king_model
from amuse.lab import nbody_system
from amuse.lab import Hermite
from amuse.plot import *
from matplotlib.pyplot import xscale, yscale

def grav_3body(t_end):
  
  N = 100
  W0 = 3
  Rinit     = 50.  | units.parsec
  timestep  = 0.01 | units.Myr
  Mcluster  = 4.e4 | units.MSun
  Rcluster  = 0.7  | units.parsec
  converter = nbody_system.nbody_to_si(Mcluster,Rcluster)

  particles = new_king_model(N, W0, convert_nbody=converter)
  particles.radius = 0.0 | units.parsec

  print("Finished generating particles!")

  cluster = Hermite(converter, parameters=[("epsilon_squared", (0.01 | units.parsec)**2)])
  cluster.particles.add_particles(particles)

  x=cluster.particles.x.value_in(units.parsec)
  y=cluster.particles.y.value_in(units.parsec)# native_plot.savefig("1.png")
  scatter(x,y)

  cluster.evolve_model(1.|units.Myr,timestep = timestep,verbose=True)

  x=cluster.particles.x.value_in(units.parsec)
  y=cluster.particles.y.value_in(units.parsec)

  scatter(x,y)
  native_plot.savefig("3.png")

  # print(particles.x)

if __name__ == "__main__":
  grav_3body(10)