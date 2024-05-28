import al26_nbody
from amuse.units import units
from amuse.lab import SeBa
from amuse.datamodel import Particles
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable


N_STARS = 1000
N_STEPS = 1000

MIN_MASS = 20
MAX_MASS = 60

plt.rcParams.update({
  "text.usetex": True,
  "font.family": "serif",
  "font.serif": ["Computer Modern Roman"],
})

def calc_total_mass_loss(masses):
  evol = SeBa(number_of_workers = 1)
  evol.parameters.metallicity = 0.02
  evol.particles.add_particles(Particles(mass = masses))

  # Work out how long to evolve system
  min_mass = min(masses)
  approx_lifespan = 2.0 * ((1e10 | units.yr) * ((1 | units.MSun) / min_mass)**2.5)
  # Evolve system
  evol.evolve_model(approx_lifespan)
  final_masses = evol.particles.mass
  m_loss_tot = masses - final_masses
  print(m_loss_tot.value_in(units.MSun))
  evol.stop()
  return m_loss_tot






SLRs = al26_nbody.read_SLRs("slr-abundances.csv")

print(SLRs["Al26"])

star_masses = np.linspace(MIN_MASS,MAX_MASS,N_STARS) | units.MSun

yields = np.zeros(N_STARS)
for i,mass in enumerate(tqdm(star_masses)):
  al26_wind_yield = al26_nbody.calc_slr_yield(mass,SLRs["Al26"].wind_mass,SLRs["Al26"].wind_yield)
  print(al26_wind_yield)
  yields[i] = al26_wind_yield.value_in(units.MSun)
total_mass_loss = calc_total_mass_loss(star_masses).value_in(units.MSun)
wind_ratios = yields / total_mass_loss

# Begin evolving cluster to find mass loss rate over time
evol = SeBa(number_of_workers=6)
evol.parameters.metallicity = 0.02
evol.particles.add_particles(Particles(mass = star_masses))
times = np.linspace(0,30,N_STEPS) | units.Myr


plot_data = np.zeros((N_STARS,N_STEPS))
m_i = evol.particles.mass.value_in(units.MSun)
for i,time in enumerate(tqdm(times)):
  evol.evolve_model(time)
  m = evol.particles.mass.value_in(units.MSun)
  for j,mass in enumerate(m):
    plot_data[j,i] = (wind_ratios[j]*(m_i[j]-m[j]))

cols = mpl.cm.inferno(np.linspace(1,0,N_STARS))

plt.figure(figsize=(7.5,3))

for i in range(N_STARS):
  t = times.value_in(units.Myr)
  y = plot_data[::-1][i]
  y_life = np.unique(y)
  t_life = np.asarray(t[0:len(y_life)])
  plt.semilogy(t_life,y_life,color=cols[i])

plt.xlabel("Age, $t$ (Myr)")
plt.ylabel("$^{26}$Al wind yield, $\\sum{\\textrm{M}_{\\textrm{26Al}}}$ (M$_\\odot$)")
plt.grid(True, which="both", ls="dotted",alpha=0.3)
# This part in particular sucks, please email me if you know a better way of bolting a color bar onto a line plot
divider = make_axes_locatable(plt.gca())
ax_cb = divider.new_horizontal(size="5%", pad=0.05)    
cbar = mpl.colorbar.ColorbarBase(ax_cb, cmap=mpl.cm.inferno, orientation='vertical',ticks=np.linspace(0,1,5))
# Hard mode: it has to use nicely formatted ticks
cticks = []
for n in np.linspace(MIN_MASS,MAX_MASS,5):
  cticks.append(int(n))
cbar.ax.set_yticklabels(cticks)
cbar.set_label("Mass, M$_\\star$ (M$_\\odot$)")
plt.gcf().add_axes(ax_cb)

plt.savefig("cumulative-yield.pdf",bbox_inches="tight")




