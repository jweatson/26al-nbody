import sys
import os
script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append(script_dir)
sys.path.append(script_dir+"/../")

import matplotlib.colors as mcolors
import matplotlib
from al26_plot import read_yields,read_state,calc_cdf,use_tex
from al26_nbody import Yields,State,Metadata
from glob import glob
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.ticker as mticker
import matplotlib.cm as cm
from tqdm import tqdm
import seaborn as sns
import pandas as pd

simsets = sorted(glob("./pt-*/pt-*/"))

nstar_arr   = []
rc_arr      = []
sim_n_arr   = []
simname_arr = []
model_arr   = []
isotope_arr = []
yields_arr  = []

for simset in simsets:
  print(simset)
  sims = sorted(glob(simset+"pt-*-*/"))
  max_al_local = []
  
  for i,sim in enumerate(sims):
    yields_fname = sorted(glob(sim+"*yields*.zst"))[-1]
    last_state_fname = sorted(glob(sim+"*-state-*.zst"))[-1]
    final_state = read_state(last_state_fname)
    cluster = final_state.cluster
    metadata = final_state.metadata

    N  = metadata.args.n
    RC = metadata.args.rc
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
    
    simulation_name = "pt-"+str(RC)+"-"+str(N)
    for model in ["local","global"]:
      for isotope in ["26al","60fe"]:
        nstar_arr.append(N)
        rc_arr.append(str(RC))
        sim_n_arr.append(i)
        simname_arr.append(simulation_name)
        model_arr.append(model)
        isotope_arr.append(isotope)

    yields_arr.append(np.max(ratio_local_26al_sne / 5.85e-5))
    yields_arr.append(np.max(ratio_local_60fe_sne / 1e-6))
    yields_arr.append(np.max(ratio_global_26al_sne / 5.85e-5))
    yields_arr.append(np.max(ratio_global_60fe_sne / 1e-6))


df = pd.DataFrame({"nstars": nstar_arr,
  "rc": rc_arr,
  "sim": sim_n_arr,
  "simname": simname_arr,
  "model": model_arr,
  "isotope": isotope_arr,
  "yield": yields_arr,
})
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.dropna(how='any')
df = df[df.rc != "10.0"]
# df = df[df.nstars != 100]

use_tex()
# plt.figure(figsize=(8,16))

# df = df[df.model == "global"]
# df = df[df.isotope == "26al"]

# sns.displot(df,x="yield",hue="simname",kind="kde")

fig,ax = plt.subplots(ncols=1,nrows=2,sharex=True,figsize=(5,5))
# sns.set_palette("pastel",10)

df2 = df[df.model == "local"]

dfal = df2[df2.isotope == "26al"]
# sns.stripplot(ax=ax[0],data=dfal,x="rc",y="yield",hue="nstars",dodge=True,palette=sns.color_palette(),linewidth=1,jitter=0.20,alpha=0.8)
sns.boxplot(ax=ax[0],data=dfal,x="rc",y="yield",hue="nstars",dodge=True,palette=sns.color_palette(),showfliers=True,linewidth=1)

dffe = df2[df2.isotope == "60fe"]
# sns.stripplot(ax=ax[1],data=dffe,x="rc",y="yield",hue="nstars",dodge=True,palette=sns.color_palette(),linewidth=1,jitter=0.20,alpha=0.8)
sns.boxplot(ax=ax[1],data=dffe,x="rc",y="yield",hue="nstars",dodge=True,palette=sns.color_palette(),showfliers=True,linewidth=1)

ax[0].set_yscale("log")
ax[1].set_yscale("log")


# ns_range = set(df.nstars)
# for i,ns in enumerate(ns_range):
#   df2 = df[df.nstars == ns]

#   print(df2.sort_values("yield"))
  

#   sns.violinplot(ax=ax[0],data=df2[df.isotope == "26al"],x="rc",y="yield",hue="model",
#                 order=["0.3","1.0","3.0"],bw_adjust=0.05,cut=0,inner=None,
#                 fill=False,split=True,linewidth=1,palette={"local":palette[i],"global":palette[i]})
#   sns.violinplot(ax=ax[1],data=df2[df.isotope == "60fe"],x="rc",y="yield",hue="model",
#                 order=["0.3","1.0","3.0"],bw_adjusst=0.05,cut=0,inner=None,
#                 fill=False,split=True,linewidth=1,palette={"local":palette[i],"global":palette[i]})
  
for axes in ax:
  axes.get_legend().remove()
  # axes.yaxis.set_major_formatter(mticker.StrMethodFormatter("$10^{{{x:.0f}}}$"))
  # ymin, ymax = axes.get_ylim()
  # tick_range = np.arange(np.floor(ymin), ymax,4)
  # axes.yaxis.set_ticks(tick_range)
  # axes.yaxis.set_ticks([np.log10(x) for p in tick_range for x in np.linspace(10 ** p, 10 ** (p + 1), 10)], minor=True)

# plt.tight_layout()


#   for collection in axes.collections:
#     if isinstance(collection, matplotlib.collections.PolyCollection):
#         collection.set_edgecolor(collection.get_facecolor())
#         collection.set_facecolor('none')

# for h in ax.legend_.legendHandles:
#     if isinstance(h, matplotlib.patches.Rectangle):
#         h.set_edgecolor(h.get_facecolor())
#         h.set_facecolor('none')
#         h.set_linewidth(1.5)

# sns.violinplot(ax=ax[1],data=df,x="Rc",y="ratio_local_60fe_sne",hue="Nstars",
#                order=["0.3","1.0","3.0","10.0"],inner="quart",
#                fill=False,split=True,linewidth=1)

ax[0].set_ylabel("$^{26}$Al enrichment, $\\Lambda_{26\\textrm{Al}}$")
ax[1].set_ylabel("$^{60}$Fe enrichment, $\\Lambda_{60\\textrm{Fe}}$")

ax[0].set_xlabel("")
ax[1].set_xlabel("Half-mass radius, R$_\\textrm{c}$ (pc)")


lines_labels = [ax[0].get_legend_handles_labels()]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels,loc="upper center",ncol=3,title="$N_{\\star}$")

# plt.yscale("log")

plt.savefig("violin.pdf",bbox_inches="tight")