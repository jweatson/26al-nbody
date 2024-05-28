
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from amuse.units import units
from amuse.lab import SeBa,Particle
from scipy.integrate import cumtrapz
from scipy.interpolate import Akima1DInterpolator
from tqdm import tqdm
import matplotlib.pylab as pl
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys

msol = units.MSun
myr  = units.Myr
yr   = units.yr

plt.rcParams.update({
  "text.usetex": True,
  "font.family": "serif",
  "font.serif": ["Computer Modern Roman"],
})

def calc_wind_ratio(mass,yields,name):
  """
  Calculate the ratio of Al26 in the wind of a massive star
  This is calculated by averaging the wind yield with the total mass loss of a star
  A 2nd order polynomial fit is used to estimate the wind yield

  INPUTS:
  mass = mass in AMUSE units
  fit  = nth order polynomial fit, as a list or array
  """
  masses = [13,15,20,25,30,40,60,80,120]
  m_msol = mass.value_in(msol)

  print("hi")
  print(type(yields))
  print(np.shape(yields))

  yyy = np.log10(yields*1.0)

  tck = Akima1DInterpolator(masses,np.log10(yields))
  wind_yield = 10**tck(m_msol)
  # print(m_msol,wind_yield)
  # xxx = np.linspace(13,120,1000)
  # yyy = splev(xxx,tck)
  # plt.semilogy(xxx,yyy)
  # plt.scatter(masses,yields)
  # plt.savefig("test.pdf")
  # sys.exit()
  # Find the total lifetime of the star, ahead of time
  evol = SeBa(number_of_workers=1)
  evol.particles.add_particle(Particle(mass=mass))
  # Evolve the star until it dies
  evol.evolve_model(30. | myr)
  # Get the stars final mass
  final_mass = evol.particles[0].mass
  # Now, calculate the total mass loss rate
  m_loss_tot = (mass - final_mass).value_in(msol)
  # wind_yield = 10**polyobj(m_msol)
  wind_ratio = wind_yield / m_loss_tot
  # print(mass.value_in(msol),final_mass.value_in(msol),m_loss_tot,wind_yield,wind_ratio)
  evol.stop()
  del evol
  # Finish!
  return wind_ratio

def yield_rate(wind_ratio,mass,times):
  m_msol = mass.value_in(msol) 
  # Find the total lifetime of the star, ahead of time
  evol = SeBa(number_of_workers=1)
  evol.particles.add_particle(Particle(mass=mass))
  # 
  t = []
  y = []
  for time in times:
    evol.evolve_model(time)
    t.append(evol.get_time().value_in(yr))
    # print(evol.get_time().value_in(yr))
    mdot = evol.particles[0].wind_mass_loss_rate
    yiel = abs(mdot * wind_ratio).value_in(msol/yr)
    y.append(yiel)
  # print(y)
  # print(max(y))
  y_int = cumtrapz(y,t,initial=0)
  return t,y_int

def main():
  nstars = 10
  maxmass = 60
  minmass = 20

  fit_data = pd.read_csv("wind-yields.csv")

  # Here is some of the worst plotting code I have ever written
  plt.figure(figsize=(5,4))
  for i,fit in fit_data.iterrows():
    name = fit.to_list()[2]
    if name == "Al26":
      fit_params = fit.to_numpy()[3:]
      masses = np.linspace(minmass,maxmass,nstars) | msol
      times  = np.linspace(0,10,10) | myr
      cols = mpl.cm.inferno(np.linspace(1,0,nstars)) # 1 to 0 as front to back, it's weird but trust me here
      # masses = 15.0 | msol

      wind_ratios = []
      for i in tqdm(range(len(masses))):
        # Lines ordered front to back, otherwise it looks messy
        mass = masses[::-1][i]
        wind_ratio = calc_wind_ratio(mass,fit_params,name)
        t,y = yield_rate(wind_ratio,mass,times)
        y_life = np.unique(y)
        t_life = np.asarray(t[0:len(y_life)])/1e6
        plt.semilogy(t_life,y_life,color=cols[i])
        # plt.ylim((1e-10,1))

      plt.xlabel("Star age, $t$ (Myr)")
      plt.ylabel("$^{26}$Al wind yield, $\\sum{\\textrm{M}_{\\textrm{26Al}}}$ (M$_\\odot$)")
      plt.grid(True, which="both", ls="dotted",alpha=0.3)
      # This part in particular sucks, please email me if you know a better way of bolting a color bar onto a line plot
      divider = make_axes_locatable(plt.gca())
      ax_cb = divider.new_horizontal(size="5%", pad=0.05)    
      cbar = mpl.colorbar.ColorbarBase(ax_cb, cmap=mpl.cm.inferno, orientation='vertical',ticks=np.linspace(0,1,5))
      # Hard mode: it has to use nicely formatted ticks
      cticks = []
      for n in np.linspace(minmass,maxmass,5):
        cticks.append(int(n))
      cbar.ax.set_yticklabels(cticks)
      cbar.set_label("Star mass, M$_\\star$ (M$_\\odot$)")
      plt.gcf().add_axes(ax_cb)
      # Let's all pretend we didn't see all of that
      plt.savefig("cumulative-yield.pdf",bbox_inches="tight")

main()