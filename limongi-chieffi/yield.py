import sys
sys.path.append("../")
from al26_nbody import Al26WindRatio
import numpy as np
import matplotlib.pyplot as plt

from amuse.lab import SeBa
from amuse.datamodel import Particles
from amuse.units import units

from scipy import integrate

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
  })
plt.figure(figsize=(5,4),dpi=600)
plt.grid(True, which="both", ls="dotted",alpha=0.3)

massr = [20,30,40,50,60,70,80] | units.MSun
nmass = len(massr)

stars = Particles(nmass)
stars.mass = massr

stellar = SeBa()
stellar.particles.add_particles(stars)

tr = np.linspace(0,10.,1000) | units.Myr
mr = []
for t in tr:
  stellar.evolve_model(t)
  mr.append(abs(stellar.particles.wind_mass_loss_rate.value_in(units.MSun/units.yr)))
mr = np.asarray(mr)

print(np.shape(mr))

for i in range(nmass):
  wr = Al26WindRatio(massr[i])
  x = tr.value_in(units.yr)
  y = mr[:,i] * wr
  y_int = integrate.cumtrapz(y, x, initial=0)

  x_plt = []
  y_plt = []
  for j in range(1,len(y_int)):
    x_plt.append(x[j]/1e6)
    y_plt.append(y_int[j])
    if y_int[j-1] == y_int[j]:
      break
  label = r"{}$\,$M$_\odot$".format(massr[i].value_in(units.MSun))
  plt.semilogy(x_plt,y_plt,label=label)
  plt.scatter(x_plt[-1],y_plt[-1],marker="x")

plt.legend()
plt.xlabel("t (Myr)")
plt.ylabel("$\sum$M$_{26Al}$ (M$_\odot$)")
plt.ylim((1e-10,2e-4))
plt.savefig("cumulative_yield.pdf",bbox_inches="tight")
plt.show()
