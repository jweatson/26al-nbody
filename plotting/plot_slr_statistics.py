import sys
import os
script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append(script_dir)
sys.path.append(script_dir+"/../")

import matplotlib.colors as mcolors
from al26_plot import read_yields,read_state,calc_cdf,use_tex
from al26_nbody import Yields,State,Metadata
from glob import glob
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.ticker as mticker

use_tex()
fig = plt.figure(figsize=(8,5))
ax = fig.subplots(1,2,sharey=True)

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
  # Calculate ratios for 26Al
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

  x_g,y_g = calc_cdf(ratio_global_26al_sne)
  x_l,y_l = calc_cdf(ratio_local_26al_sne)
  ax[0].plot(x_g,y_g,c=color)
  ax[0].plot(x_l,y_l,c=color,linestyle="dashed")

  x_g,y_g = calc_cdf(ratio_global_60fe_sne)
  x_l,y_l = calc_cdf(ratio_local_60fe_sne)
  ax[1].plot(x_g,y_g,c=color)
  ax[1].plot(x_l,y_l,c=color,linestyle="dashed")

  print(max(ratio_global_60fe_sne))
  print(max(ratio_local_60fe_sne))
  print(max(ratio_global_60fe))
  print(max(ratio_local_60fe))
  # plt.semilogx(base_global[:-1], cumulative_global, c=color)
  # plt.semilogx(base_local[:-1],cumulative_local,c=color,linestyle="dotted")

for a in ax:
  a.set_ylim(0,1)
  a.set_xscale("log")
  a.xaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs="auto"))
  a.grid(True,which="both",linestyle=":",alpha=0.3)


ax[0].set_xlim(1e-10,1e-2)
ax[0].axvline(x=5.85e-5,c="k",linestyle="dotted")
ax[0].set_xlabel("$^{26}$Al/$^{27}$Al")
ax[0].set_ylabel("CDF")

ax[1].set_xlim(1e-10,1e-4)
ax[1].axvline(x=1e-6,c="k",linestyle="dotted")
ax[1].set_xlabel("$^{60}$Fe/$^{56}$Fe")

plt.savefig("cdf.pdf",bbox_inches="tight")

# yields.ratio_al_local = yields.local_26al / 
