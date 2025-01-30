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
import matplotlib.cm as cm
from tqdm import tqdm

try:
  outname = sys.argv[1]
except:
  outname = "cdf"

use_tex()
fig  = plt.figure(figsize=(6,6))
axes = fig.subplots(2,2,sharey=True,sharex=True)
sims = sorted(glob("./*/"))

cmaps = np.linspace(0,1,len(sims))

for i,sim in enumerate(tqdm(sims)):
  color = cm.get_cmap("GnBu")(cmaps[i])
  # Get filename yields
  yields_fname = sorted(glob(sim+"*yields*.zst"))[-1]
  # Read in yields
  yields = read_yields(yields_fname)
  # Get filename for most recent yields file
  last_state_fname = sorted(glob(sim+"*-state-*.zst"))[-1]
  # Read in state, extract cluster data
  final_state = read_state(last_state_fname)
  cluster = final_state.cluster
  nstars = len(cluster.mass)
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

  xgal,ygal = calc_cdf(ratio_global_26al_sne)
  xgfe,ygfe = calc_cdf(ratio_global_60fe_sne)
  xlal,ylal = calc_cdf(ratio_local_26al_sne)
  xlfe,ylfe = calc_cdf(ratio_local_60fe_sne)

  axes[0,0].plot(xgal,ygal,c=color)
  axes[0,1].plot(xgfe,ygfe,c=color)
  axes[1,0].plot(xlal,ylal,c=color,linestyle="dashed")
  axes[1,1].plot(xlfe,ylfe,c=color,linestyle="dashed")

for i in range(len(axes)):
  for j in range(len(axes[0])):
    axes[i,j].set_xscale("log")

    # axes[i,j].set_yscale("log")
    # ymin = 10**(np.floor(np.log10(1/nstars)))

    axes[i,j].set_ylim(0,1)
    axes[i,j].set_xlim(1e-12,1e-2)
    axes[i,j].xaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs="auto"))
    axes[i,j].grid(True,which="both",linestyle=":",alpha=0.3)

for i in range(len(axes)):
  axes[i,0].set_ylabel("CDF")
  axes[i,0].axvline(x=5.85e-5,c="k",linestyle="dotted")
for i in range(len(axes)):
  axes[i,1].axvline(x=1e-6,c="k",linestyle="dotted")

axes[1,0].set_xlabel("$^{26}$Al/$^{27}$Al")
axes[1,1].set_xlabel("$^{60}$Fe/$^{56}$Fe")

axes[0,0].set_title("$^{26}$Al global model")
axes[0,1].set_title("$^{60}$Fe global model")
axes[1,0].set_title("$^{26}$Al local model")
axes[1,1].set_title("$^{60}$Fe local model")

# for axin axes:
  # ax.set_ylim(0,1)
  # ax.set_xscale("log")


# axes[0].set_xlim(1e-10,1e-2)
# axes[0].set_ylabel("CDF")

# axes[1].set_xlim(1e-10,1e-4)
# axes[1].axvline(x=1e-6,c="k",linestyle="dotted")

plt.savefig(outname+".pdf",bbox_inches="tight")

# yields.ratio_al_local = yields.local_26al / 
