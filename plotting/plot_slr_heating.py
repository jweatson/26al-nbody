import sys
import os
script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append(script_dir)
sys.path.append(script_dir+"/../")

import matplotlib.colors as mcolors
from al26_plot import read_yields,read_state,calc_cdf,use_tex,calc_current_heating_rate
from al26_nbody import Yields,State,Metadata
from glob import glob
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.ticker as mticker

from amuse.units import units

use_tex()
fig = plt.figure(figsize=(8,3))
axes = fig.subplots(1,1)

sims = sorted(glob("*/"))
for i,sim in enumerate(sims):
  print(sim)
  color = list(mcolors.TABLEAU_COLORS)[i]
  # Get filename yields
  yields_fname = sorted(glob(sim+"*yields*.zst"))[-1]
  # Read in yields
  yields = read_yields(yields_fname)
  # Get filename for most recent yields file
  last_state_fname = sorted(glob(sim+"*-state-*.zst"))[-1]
  # Read in state, extract cluster data
  final_state = read_state(last_state_fname)
  cluster = final_state.cluster

  ratio_local_26al = cluster.mass_26al_local / cluster.mass_27al
  ratio_global_26al = cluster.mass_26al_global / cluster.mass_27al
  ratio_sne_26al = cluster.mass_26al_sne / cluster.mass_27al
  ratio_local_26al_sne = ratio_local_26al + ratio_sne_26al
  ratio_global_26al_sne = ratio_global_26al + ratio_sne_26al
  # Calculate ratios for 60Fe
  ratio_local_60fe = cluster.mass_60fe_local / cluster.mass_56fe
  ratio_global_60fe = cluster.mass_60fe_global / cluster.mass_56fe
  ratio_sne_60fe = cluster.mass_60fe_sne / cluster.mass_56fe
  ratio_local_60fe_sne = ratio_local_60fe + ratio_sne_60fe
  ratio_global_60fe_sne = ratio_global_60fe + ratio_sne_60fe

  Q_global = calc_current_heating_rate(ratio_global_26al_sne,ratio_global_60fe_sne)
  Q_local  = calc_current_heating_rate(ratio_local_26al_sne,ratio_local_60fe_sne)

  x_g,y_g = calc_cdf(Q_global)
  x_l,y_l = calc_cdf(Q_local)

  axes.plot(x_g,y_g,c=color)
  axes.plot(x_l,y_l,c=color,linestyle="dashed")

axes.set_xlim(1e-14,1e-7)
axes.axvline(x=5.85e-5,c="k",linestyle="dotted")
axes.set_xlabel("$^{26}$Al/$^{27}$Al")
axes.set_ylabel("CDF")
axes.set_xlabel("$^{60}$Fe/$^{56}$Fe")
axes.set_xscale("log")


qss = 3.4e-6

axes.axvline(x=qss,c="k",linestyle="dotted")
plt.savefig("slr_heating.pdf")

