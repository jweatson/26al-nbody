"""
Postprocessing script for al26 nbody paper, will postprocess an entire folder into a single large compressed pandas set for plotting out data

TODO:
- Import folder, read all filenames
- Copy data to pandas array, use lists for appending and reprocessing data, though this is very clunky
  - Might be better to use a dictionary instead of individual discrete arrays?
- Handle decay, function needs to be written in al26_plot

NOTES:
I am truly sorry, some of the code I have written here is either really inefficient or weirdly written
Ideally you want everything in a pandas file, but some of the postprocessing requires really inefficient steps/can't be easily vectorised or passed through to numpy instructions
As such, I will probably rewrite the al26_nbody simulation code itself to do the role of this postprocessor on the fly
"""

import pickle
import ubjson
import zstandard as zstd
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import numba as nb
from glob import glob

import sys,os
plotting_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
al26_nbody_dir = plotting_dir+"/../"
sys.path.append(al26_nbody_dir)
sys.path.append(al26_nbody_dir+"/plotting/")

from al26_nbody import State,Metadata,Yields,myr,pc,msol
import scipy
from amuse.units import units

from al26_plot import calc_disk_final_enrichment,read_yields,read_state

import pandas as pd

# Properties to read in 
properties = ["26al_local","26al_global","26al_sne","60fe_local","60fe_global","60fe_sne"]
columns = [""]

# First, build the dictionary (bunch of empty lists)
proc_data = {}
proc_data["nstars"] = []
proc_data["rc"] = []
proc_data["sim_number"] = []
proc_data["star"] = []
proc_data["initial_mass"] = []
proc_data["mass"] = []
proc_data["isotope"] = []
proc_data["model"] = []
proc_data["yield_ratio_nodecay"] = []
proc_data["yield_ratio_decay"] = []

# proc_data["nstars","rc","sim_number","model","isotope","yield"]

# Get full list of simulation sets
simsets = sorted(glob("./pt-**/pt*/"))


tt1 = tqdm(simsets,position=0)
for simset in tt1:
  tt1.set_description(simset)
  sims = sorted(glob(simset+"pt-*/"))
  tt2  = tqdm(sims,position=1,leave=False)
  for sim_number,sim in enumerate(tt2):
    yields_fname = sorted(glob(sim+"*yields*ubj.zst"))[-1]
    state_fnames = sorted(glob(sim+"*-state-*.zst"))
    last_state_fname = state_fnames[-1]
    first_state_fname = state_fnames[0]
    sim_yield = read_yields(yields_fname)
    final_state = read_state(last_state_fname)
    metadata = final_state.metadata
    cluster  = final_state.cluster
    # Draw specific attributes for postprocessing
    nstars = metadata.args.n
    rc = metadata.args.rc
    lifetimes = cluster.tau_disk.value_in(myr)
    # Determine final values for each star in the case of decaying disks (achieved through interpolation)
    sim_yield = calc_disk_final_enrichment(sim_yield,lifetimes)
    # Get initial masses
    first_state = read_state(first_state_fname)
    initial_masses = first_state.cluster.mass

    isotopes        = ["26al","60fe"]
    stable_isotopes = ["27al","56fe"]
    models          = ["local","global","sne","local+sne","global+sne"]

    for star_number,star in enumerate(cluster):
      mass = star.mass.value_in(msol)
      initial_mass = initial_masses[star_number].value_in(msol)
      for iso_index,isotope in enumerate(isotopes):
        for model in models:
          stable_parameter_name   = "mass_"+stable_isotopes[iso_index]
          submodels = model.split("+")

          stable_yield = getattr(star,stable_parameter_name)
          unstable_yield_nodecay = 0.0 | units.kg
          unstable_yield_decay   = 0.0 | units.kg
          # Add together submodels if needed, get from yields
          for submodel in submodels:
            unstable_parameter_name = submodel+"_"+isotope
            nodecay = getattr(sim_yield,unstable_parameter_name)[-1,star_number] | msol
            decay   = getattr(sim_yield,unstable_parameter_name+"_final")[star_number] | msol
            unstable_yield_nodecay += nodecay
            unstable_yield_decay   += decay

          proc_data["nstars"].append(nstars)
          proc_data["rc"].append(rc)
          proc_data["sim_number"].append(sim_number)
          proc_data["star"].append(star_number)
          proc_data["initial_mass"].append(initial_mass)
          proc_data["mass"].append(mass)
          proc_data["isotope"].append(isotope)
          proc_data["model"].append(model)
          proc_data["yield_ratio_nodecay"].append(unstable_yield_nodecay / stable_yield)
          proc_data["yield_ratio_decay"].append(unstable_yield_decay / stable_yield)

df = pd.DataFrame.from_dict(proc_data)
df.to_pickle("all-sims-ratios.pkl.zst")
print("Finished processing!")