"""
26al-nbody.py
- This programme is meant to run as a standalone application, but the functions can be spun out!
- This is a "spiritual successor" to the work by: Lichtenberg, T., Parker, R.J. & Meyer, M.R. (2016)
  which covers 60Fe and 26Al enrichment of protostellar disks by supernoava injection
    Lichtenberg, T., Parker, R. J., & Meyer, M. R. (2016).
    Isotopic enrichment of forming planetary systems from supernova pollution.
    Monthly Notices of the Royal Astronomical Society, 462, 3979–3992.
    https://doi.org/10.1093/mnras/stw1929
- This is meant as both a test of the AMUSE library as well as the first steps towards a simulation
  on deposition of SLRs onto protstellar disks
"""

### LIBRARIES
# Standard python libraries
import argparse # Programme arguments
import sys      # Paths
# Amuse imports, complex library so only importing specific things
from amuse.units import units # Unit conversion
from amuse.ic import plummer # Cluster model
from amuse.community.fractalcluster.interface import new_fractal_cluster_model
from amuse.lab import nbody_system, Hermite, SeBa, Particle # Simulation libraries
from amuse.community.bhtree import Bhtree
from amuse.community.ph4 import Ph4
# Plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns
# Math/Array libraries
from random import uniform
import numpy as np # Numpy library, what code isn't complete without it?
import numba as nb # Numba JIT library, used for accelerating less performant parts of the code
import pandas as pd # Pandas table library, used for some data handling
from numpy.random import uniform, exponential # Random distributions for IMF 
from math import exp, ceil, pi # Always useful
from scipy.interpolate import splev,splrep,Akima1DInterpolator # Interpolation libraries for wind yield calculations
# File saving
import re
import os.path
from glob import glob
import pickle # Pickle is used to store the state object, which is used for checkpoints
import ubjson # UBJSON is used to store the yields object (less versatile, much much faster)
import zstandard as zstd # ZSTD (https://facebook.github.io/zstd/) is used for compression
# Others
from tqdm import tqdm # Good little progress bar
from time import time
from datetime import datetime
# Warnings for numba
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

# GLOBAL VALUES
n_plot           = 100 # Number of checkpoints to make
steps_per_plot   = 10  # Number of substeps to make per write
module_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
workers          = 8
### MODELS
gravity_model = "bhtree" 
stellar_model = SeBa    # Using SeBa as it is fast and relatively accurate

# Declare units in global namespace, using amuse units, this just saves a lot of time and space
# Mass units
kg     = units.kg       # Kilogram, 1.0e+3 g (obviously)
msol   = units.MSun     # Solar mass, 1.9884099e+33 g
# Time units
yr     = units.yr       # 1 standard year, 3.1556926e+7 s
myr    = units.Myr      # Million years, 3.1556926e+13 s
# Distance units
au     = units.au       # Astronomical unit, 1.4959787e+13 cm
pc     = units.parsec   # Parsec, 3.0856776e+18 cm
# Composite units
msolyr = msol * yr**-1  # Solarmass per year, 6.3010252e+25 g/s
kms    = units.kms      # Kilometers per second, 1.0e+6 cm/s

# SIMULATION VALUES
r_bub_local_wind = 0.1 | pc # Size of bubble interation region for wind 
r_bub_local_sne  = 1.0 | pc # Size of bubble interaction region for supernovae
AGBs = None

# FILE HANDLING ROUTINES AND CLASSES

class Metadata:
  def __init__(self,args,t_f) -> None:
    # Initialisation time
    self.sim_start     = datetime.now()
    self.sim_start_str = self.sim_start.strftime("%d/%m/%Y %H:%M:%S")
    # Last run time
    self.update_access_time()
    # All arguments
    self.args          = args # Import arguments from initialisation
    # Initial arguments
    self.model          = args.model
    self.nstars         = args.n
    self.cluster_radius = args.rc
    # Store or generate base filename for all files in the simulation
    if args.filename == "":
      self.filename = self.generate_filename()
    else:
      self.filename = args.filename
    # Time of simulation
    self.time       = 0.0 | myr
    self.t_f        = t_f
    self.completion = 0.0 # Fraction of simulation completed
    self.most_recent_checkpoint = 0
  def generate_filename(self):
    """
    If no base filename is present, generate a base name
    """
    time_string = self.sim_start.strftime("%Y-%m-%d-%H-%M-%S")
    filename = "sim-"+time_string
    return filename
  def update(self,current_time,increment_checkpoint=True):
    if increment_checkpoint:
      self.most_recent_checkpoint += 1
    self.update_completion(current_time)
    self.update_access_time()
  def update_completion(self,current_time):
    self.time = current_time
    self.completion = self.time / self.t_f
  def update_access_time(self):
    self.sim_last     = datetime.now()
    self.sim_last_str = self.sim_last.strftime("%d/%m/%Y %H:%M:%S")

class Yields():
  def __init__(self,filename) -> None:
    """
    Initialisation function for the class, creates a series of empty arrays and stores the filename, provided as an argument
    """
    # Initialise each array, each SLR needs both global and local model storage
    ## 26Al
    self.filename          = filename
    self.time              = []
    # 26al Masses assuming no disk progression
    self.local_26al        = []
    self.global_26al       = []
    self.sne_26al          = []
    self.agb_26al          = []
    self.agb_26al_raw      = []
    # 26al Summations
    self.sum_local_26al    = []
    self.sum_global_26al   = []
    self.sum_sne_26al      = []
    self.sum_agb_26al      = []
    # 60fe Masses
    self.local_60fe        = []
    self.global_60fe       = []
    self.sne_60fe          = []
    self.agb_60fe          = []
    self.agb_60fe_raw      = []
    # 60fe Summations
    self.sum_local_60fe    = []
    self.sum_global_60fe   = []
    self.sum_sne_60fe      = []
    self.sum_agb_60fe      = []
    # Final values assuming disk progression
    # 26Al
    self.local_26al_final  = []
    self.global_26al_final = []
    self.sne_26al_final    = []
    self.agb_26al_final    = []
    # 60Fe
    self.local_60fe_final  = []
    self.global_60fe_final = []
    self.sne_60fe_final    = []
    self.agb_60fe_final    = []
    self.first_write       = True
    return 
  def update_state(self,model_time,cluster):
    """
    Update the general state of the yields object, this should be done at the end of every timestep in the execution loop
    Appended lists are used as the data structure for this as appending a python list has a complexity of O(1), whereas numpy arrays or pandas arrays have a complexity of O(n) or worse.
    Keep in mind that as this appends this will result in memory being slowly accrued by the simulation, so it might be a good idea to check this if the simulation keeps crashing.
    Inputs:
      - model_time: Simulation time in AMUSE units (converted to myr in datatype)
      - cluster: cluster object to read from
    """
    # Update contents of yields object
    self.time.append(model_time.value_in(myr))
    self.local_26al.append(list(cluster.mass_26al_local.value_in(msol)))
    self.global_26al.append(list(cluster.mass_26al_global.value_in(msol)))
    self.sne_26al.append(list(cluster.mass_26al_sne.value_in(msol)))
    self.agb_26al.append(list(cluster.mass_26al_agb.value_in(msol)))
    self.agb_26al_raw.append(list(cluster.mass_26al_agb_raw.value_in(msol)))
    # 26Al sums
    self.sum_local_26al.append(sum(list(cluster.mass_26al_local.value_in(msol))))
    self.sum_global_26al.append(sum(list(cluster.mass_26al_global.value_in(msol))))
    self.sum_sne_26al.append(sum(list(cluster.mass_26al_sne.value_in(msol))))
    self.sum_agb_26al.append(sum(list(cluster.mass_26al_agb.value_in(msol))))
    # 60Fe Continuous values
    self.local_60fe.append(list(cluster.mass_60fe_local.value_in(msol)))
    self.global_60fe.append(list(cluster.mass_60fe_global.value_in(msol)))
    self.sne_60fe.append(list(cluster.mass_60fe_sne.value_in(msol)))
    self.agb_60fe.append(list(cluster.mass_60fe_agb.value_in(msol)))
    self.agb_60fe_raw.append(list(cluster.mass_60fe_agb_raw.value_in(msol)))
    # 60Fe sums
    self.sum_local_60fe.append(sum(list(cluster.mass_60fe_local.value_in(msol))))
    self.sum_global_60fe.append(sum(list(cluster.mass_60fe_global.value_in(msol))))
    self.sum_sne_60fe.append(sum(list(cluster.mass_60fe_sne.value_in(msol))))
    self.sum_agb_60fe.append(sum(list(cluster.mass_60fe_agb.value_in(msol))))
    # Additionally, copy the current instance of the `final` yields
    # 26Al
    self.local_26al_final  = list(cluster.mass_26al_local_final.value_in(msol))
    self.global_26al_final = list(cluster.mass_26al_global_final.value_in(msol))
    self.sne_26al_final    = list(cluster.mass_26al_sne_final.value_in(msol))
    self.agb_26al_final    = list(cluster.mass_26al_agb_final.value_in(msol))
    # 60Fe
    self.local_60fe_final  = list(cluster.mass_60fe_local_final.value_in(msol))
    self.global_60fe_final = list(cluster.mass_60fe_global_final.value_in(msol))
    self.sne_60fe_final    = list(cluster.mass_60fe_sne_final.value_in(msol))
    self.agb_60fe_final    = list(cluster.mass_60fe_agb_final.value_in(msol))
    # You need to write something to this to fix edge case where disk has not decayed, this should be doable

    # Write CSV header first, this is written in such a way that restoring the yields object should not cause this to rewrite
    if self.first_write == True:
      self.write_csv_header()
      self.first_write = False
    # Write global properties to CSV file
    self.write_to_csv()
    return
  def write_csv_header(self):
    """
    A quick function to write the header of the CSV file upon initalisation
    Calling this function erases the contents of the cluster-yields file, so DO NOT use it more than once
    """
    with open("{}-cluster-yields.csv".format(self.filename),"w") as f:
      f.write("time,local_26al,global_26al,sne_26al,local_60fe,global_60fe,sne_60fe\n")
  def write_to_csv(self):
    """
    Append to global yields file, this is done every step
    """
    with open("{}-cluster-yields.csv".format(self.filename),"a") as f:
      f.write("{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e}\n".format(
        self.time[-1],
        self.sum_local_26al[-1],
        self.sum_global_26al[-1],
        self.sum_sne_26al[-1],
        self.sum_local_60fe[-1],
        self.sum_global_60fe[-1],
        self.sum_sne_60fe[-1]
      )) 
  def marinate(self,filename):
    """
    Convert contents of object into a serialised format for saving
    Why not just use pickle? Because it's slow and memory inefficient
    This dictionary can then be serialised using UBJSON (fastest) and compressed
    All filenames are a laboured joke that seemed funny to me at the time, thankfully its a very short function
    Strangely, Marshal doesn't work, which is slightly faster but produces byte strings, which are slow to recover
    
    Quick benchmark of a 1000 star cluster with 1000 datapoints:
    Pickle & compress: 5.3 seconds
    UBJSON & compress: ~0.2 seconds
    Inputs:
      str filename: Filename to write to, this is separate from the self.filename argument as it is not the base name
    """
    marinade = {}
    for attr, value in self.__dict__.items():
      marinade[attr] = value
    # Write serialised bytes to disk
    with open(filename, "wb") as f:
      # Serialise and compress
      marinate = compress(ubjson.dumpb(marinade))
      # Write to disk
      f.write(marinate)
  def plate(self,filename):
    """
    Read back files, the inverse of marinate
    Input:
      str filename: Filename to read
    """
    # Read in file and decompress data
    with open(filename,"rb") as f:
      compressed_data = f.read() 
      decompressed_data = decompress(compressed_data)
      preserve = ubjson.loadb(decompressed_data)
    # Copy all attributes from preserve into object self
    for attr, value in self.__dict__.items():
      self.__dict__[attr] = preserve[attr]
    return

class State():
  """
  This is a bit of a weird one, basically in order to pickle up the relevant files to checkpoint our data into a single file, this class contains 3 separate classes:
  - cluster: The positions and properties of all stars in the simulation
  - converter: A unit converter used by AMUSE
  - metadata: All of the odds and ends, see the Metadata class for more info
  This class exists only to bind together these things and store serialised in a pickle file. see save_checkpoint and load_checkpoint for the main places it is used.
  """
  def __init__(self,cluster,converter,metadata) -> None:
    self.cluster   = cluster
    self.converter = converter
    self.metadata  = metadata
    return

def most_recent_checkpoint(filename):
  """
  Use glob and regex to find the most recent checkpoint file
  Inputs:
    str filename: base filename to consider
  Outputs:
    int highest_filename: highest file number in checkpoint file series
  """
  state_string = filename+"-state-*"
  files = glob(state_string)
  regex = re.compile(r'\d+')
  highest_filenum = 0
  for file in files:
    filenum = int(regex.search(file).group(0))
    if highest_filenum < filenum:
      highest_filenum = filenum
  # Sanity check to see if file exists
  most_recent_state_filename = filename+"-state-"+str(highest_filenum).zfill(5)+".pkl.zst"
  file_exists = os.path.isfile(most_recent_state_filename)
  if file_exists:
    print("Found most recent checkpoint file: "+most_recent_state_filename)
    return highest_filenum
  else:
    raise IOError("Missing file! Somethings up!")

def compress(decompressed_data,level=8,threads=-1):
  """
  Compress bytes, not a stream compression function, this is used for a couple of save function.
  zstd (https://facebook.github.io/zstd/) is used because it's got the best performance, it's the only thing Facebook has ever made that's good.
  Inputs:
    bytes decompressed_data: Data to compress
    int   level: Compression level to use, 8 is a good tradeoff of speed and performance
    int   threads: Number of compression threads to use, 0 disables, -1 uses all available threads
  Outputs:
    bytes compressed_data: You will never guess what this is.
  """
  c = zstd.ZstdCompressor(threads=threads,level=level)
  compressed_data = c.compress(decompressed_data)
  return compressed_data

def decompress(compressed_data):
  """
  Decompress bytes, counterpart to compress, this is a wrapper.
  Inputs:
    bytes compressed_data: It's data! That's compressed!
  Outputs:
    bytes decompresed_data: Holy shit! The data is decompressed!
  """
  c = zstd.ZstdDecompressor()
  decompressed_data = c.decompress(compressed_data)
  return decompressed_data

def save_checkpoint(filename,nfile,cluster,converter,yields,metadata,bar=None):
  """
  Write state of simulation to ZSTD compressed checkpoint file and yields to ZSTD compressed file
  Serialised datatypes are used so they are better to work with and minimal file handling code has to be written in future.
  Two different types of serialisation are used:
    - Pickle is used for the state object, as it contains AMUSE datatypes, which don't work with UBJSON
    - UBJSON is used for the yields object as it is multiple orders of magnitudes faster and uses far less working memory
  After being serialised, ZSTD compression is used, as it's fast and efficient.
  Inputs:
    str filename: Base filename of checkpoint file, do _not_ add the pklxz extension!
    obj cluster:  Cluster object
    obj yields:   Yields object, pandas datafile containing current yields for each star
    obj metadata: Metadata object, containing useful simulation information, such as initialisation parameters
    int nfile:    File number, for saving state file, should increment, but does not automatically
  Outputs:
    Nothing, purely handles I/O
  """

  state_filename = filename+"-state-"+str(nfile).zfill(5)+".pkl.zst"
  yields_filename = filename+"-yields.ubj.zst"

  if metadata.args.verbose:
    if bar is None:
      print("! Saving checkpoint file \"{}\"...".format(state_filename),end=" ")

  t1 = time()
  # Pack all relevant structures into the state object
  state = State(cluster,converter,metadata) # Build state object
  # Initialise LZMA file and dump state into file
  with open(state_filename, "wb") as f:
    compressed_state = compress(pickle.dumps(state))
    f.write(compressed_state)
  # Done!
  t2 = time()

  if metadata.args.verbose:
    if bar is None:
      print("Done! Took {:3f} seconds!".format(t2-t1))

  # Write yields object to disk
  if metadata.args.verbose:
    if bar is None:
      print("! Saving yields file \"{}\"...".format(yields_filename),end=" ")

  t3 = time()
  yields.marinate(yields_filename)
  # Done!
  t4 = time()

  if metadata.args.verbose:
    if bar is None:
      print("Done! Took {:3f} seconds!".format(t4-t3))
    else:
      bar.write("Saving checkpoint #{}... Done! Took {:3f} seconds!".format(str(nfile).zfill(5),t4-t1))
  return

def load_checkpoint(filename,nfile):
  """
  Load checkpoint and yields from compressed files
  Different forms of serialisation are used as state object is very complex and won't compress using UBJSON (as it contains AMUSE datatypes), instead we use pickle, there is a minimal speed difference.
  Inputs:
    str filename: Filename of checkpoint file
  Outputs:
    obj cluster:   Cluster object
    obj yields:    Yields object, pandas datafile containing current yields for each star
    obj metadata:  Metadata object, containing useful simulation information, such as initialisation parameters
    obj converter: Unit converter object, required for initialising gravity model
  """
  # Define some filenames, fun stuff like that
  state_filename = filename+"-state-"+str(nfile).zfill(5)+".pkl.zst"
  yields_filename = filename+"-yields.ubj.zst"

  # Load in the state file
  print("! Loading state file \"{}\"...".format(state_filename),end=" ")
  t1 = time()
  with open(state_filename, "rb") as f:
    state = pickle.loads(decompress(f.read()))
    cluster = state.cluster
    converter = state.converter
    metadata  = state.metadata
  t2 = time()
  print("Done! Took {:.3f} seconds!".format(t2-t1))

  # Load in yields file
  print("! Loading yields file \"{}\"...".format(yields_filename),end=" ")
  t3 = time()
  # Decompress and restore yields file
  yields = Yields(filename) # First, create the yields object to load data into 
  yields.plate(yields_filename) # Run function in object that loads yields file to memory
  t4 = time()
  print("Done! Took {:.3f} seconds!".format(t4-t3))
  # Done! Enjoy some outputs
  return cluster,converter,yields,metadata

def calc_wind_ratio(total_wind_loss,slr_wind_yield):
  return slr_wind_yield / total_wind_loss

def calc_slr_yield(mass,masses,yields):
  """
  Calculate the total yield for both wind and supernovae deposition from a massive star (in solar masses)
  inputs:
    mass: mass of star
    masses: masses in SLR class
    yields: yields in SLR class
  """

  # print(masses)

  mass_msol   = mass.value_in(msol)   # Convert star mass to float in solar mass
  masses_msol = masses.value_in(msol) # Convert masses to floats in solar masses
  yields_msol = yields.value_in(msol) # Convert yields to floats in solar masses

  # First check for minimum and maximum mass conditions for set
  if mass_msol < min(masses_msol) or mass_msol > max(masses_msol):
    slr_yield = 0.0 | msol
  else:
    interp = Akima1DInterpolator(masses_msol,np.log10(yields_msol))
    slr_yield = float(10**interp(mass_msol)) | msol
  return slr_yield  

def calc_total_mass_loss(mass,z=0.02):
  """
  Calculate the total mass loss from a massive star in a quick way, this is used in this programme to compare the total mass loss yield, 

  Implementation:
  - Spawn an instance of SeBa with a single star with an initial mass from the function argument
  - Evolve the star to twice its estimated age, to ensure it's off the main sequence
  - Calculate the difference between the initial and final mass
  Notes:
  - Since an instance of SeBa has been set up to handle this, it is not a good idea to generate mass losses for every star, as it will take a very long time to do so
  - There are ways of speeding this up but it only has to be run a handful of times per simulation at the start)
  - This is also not designed for small stars, and is best used for massive stars (M>12Msol)
  """
  approx_lifespan = ((1e10 | yr) * ((1 | msol) / mass)**2.5) # Estimate the lifespan of the star
  simulation_time = 2.0 * approx_lifespan # Add some wiggle room
  evol = SeBa(number_of_workers=1) # Initialise stellar evolution code
  evol.parameters.metallicity = z # Set metallicity
  evol.particles.add_particle(Particle(mass=mass)) # Add a single star to code
  evol.evolve_model(simulation_time)
  final_mass = evol.particles[0].mass
  m_loss_tot = (mass - final_mass) # Total mass loss from winds

  # Cleanup
  evol.stop()
  del evol
  # Return
  return m_loss_tot

def calc_SLR_mass_loss(mass_loss_rate,wind_ratio):
  """
  Calculate the wind mass loss rate from the
  """
  return mass_loss_rate * wind_ratio

def read_AGBs():
  class AGB:
    def __init__(self,filename) -> None:
      """
      Initialise, read in data, add to datatypes
      """
      # Read in data
      data = pd.read_csv(filename)
      self.data = data
      # First, get data
      for column in list(data.columns):
        setattr(self,column,data[column].to_numpy())
      # Then convert data to amuse units
      for key in self.__dict__.keys():
        if key == "t":
          setattr(self,key,getattr(self,key) | myr)
        if key == "star_mass":
          setattr(self,key,getattr(self,key) | msol)
        if "mass_loss_rate" in key:
          setattr(self,key,getattr(self,key) | msolyr)
        if "total_mass_loss" in key:
          setattr(self,key,getattr(self,key) | msol)
  
      self.t_f = self.t[-1]

      # Determine mass from filename (only number should be AGB mass in solar masses)
      filename_rel   = filename.split("/")[-1]
      filename_under = filename_rel.split("_")
      for check in filename_under:
        try:
          self.mass = float(check) | msol
        except:
          pass

    def interp_value(self,quantity,time):
      t = self.t.value_in(myr)
      time_myr = time.value_in(myr)
      y = getattr(self,quantity)
      units = y.unit
      # Skip values if time out of range, assume zero and return
      if time_myr < t[0]: 
        return 0.0 | units
      if time_myr > t[-1]:
        return 0.0 | units
      # # Some cubic splines can misbehave if 
      # if y.max() / y.min() >= 100:
      #   uselog = True
      # else:
      #   uselog = False
      # break into base units, otherwise things get weird
      y = y.value_in(units)
      # If using log scale, convert values
      # if uselog:
      #   y = np.log10(y)
      interp = Akima1DInterpolator(t,y)
      yy = interp(time_myr)
      # print(yy)
      # Convert log space result back to linear
      # if uselog:
      #   yy = float(10.0**yy)
      yy = yy | units
      return yy
  
  filenames = glob(module_directory+"/agb_wind/agb_slr*.csv")
  AGBs = []
  for filename in filenames:
    AGBs.append(AGB(filename))
  return AGBs



def read_SLRs(filename):
  """
  Read SLRs and produce a dictionary containing instances of the SLR class
  """
  class SLR:
    def __init__(self,data) -> None:
      """
      Initialises the class and ammends the properties of the particular SLR to the class
      """
      self.name       = data[0] # SLR name
      self.daughter   = data[1] # Daughter particle name
      self.stable     = data[2] # Stable particle name
      self.half_life  = float(data[3]) | myr # Half life in megayears
      self.tau        = float(data[4]) | myr # Decay characteristic time
      self.Zss        = float(data[5])       # Solar system abundance 
      self.Zss_err    = float(data[6])       # Error in solar system abundance
      self.wind_mass  = []
      self.wind_yield = []
      self.sne_mass   = []
      self.sne_yield  = []
      return
    
  SLRs = {}
  with open(filename) as f:
    next(f)
    for line in f:
      data = line.strip().split(",")
      slr_name = data[0]
      slr_line = SLR(data)
      SLRs[slr_name] = slr_line
  
  # read_yield("limongi-chieffi-2018/wind-yields.csv")
  # print(module_directory)

  try:
    with open(module_directory+"/limongi-chieffi-2018/wind-yields.csv") as f:
        masses = f.readline().strip().split(",")[3:]
        m = []
        for mass in masses:
          m.append(float(mass[0:-1]))
        for line in f:
          data = line.strip().split(",")
          iso = data[2]
          if iso in list(SLRs.keys()):
            x = m | msol # See this is fine, because it's already floats, turns out you can have string units in amuse
            y = [float(i) for i in data[3:]] | msol # Good lord, this took a while to work out
            SLRs[iso].wind_mass = x
            SLRs[iso].wind_yield = y
  except IOError:
    raise IOError("Cannot read file wind-yields.csv")
  
  try:
    with open(module_directory+"/limongi-chieffi-2018/sne-yields.csv") as f:
      masses = f.readline().strip().split(",")[3:]
      m = []
      for mass in masses:
        m.append(float(mass[0:-1]))
      for line in f:
        data = line.strip().split(",")
        iso = data[2]
        if iso in list(SLRs.keys()):
          x = np.asarray(m) | msol
          y = [float(i) for i in data[3:]] | msol
          SLRs[iso].sne_mass = x
          SLRs[iso].sne_yield = y
  except IOError:
    raise IOError("Cannot read file sne-yields.csv")
  # Finish up!
  return SLRs

@nb.njit(parallel=True)
def calc_wind_abs(lm_id_arr,hm_id_arr,
                  x_arr,y_arr,z_arr,
                  vx_arr,vy_arr,vz_arr,
                  mdot_arr,
                  wind_ratio_arr,
                  rdisk_arr,
                  distance_limit,
                  bubble_radius,
                  dt,):
  """
  Description:
    This code is written in a particular way   
  
    Note: previously this was written in pure python, it was a lot nicer to look at, and significantly less jank. Unfortunately python is so painfully slow at this kind of looping thing, and was slower than the _N-body solver_. So it now runs in njit.

  Inputs:
    arr[int]   lm_id_arr: Array of index locations in all star array for low mass stars
    arr[int]   hm_id_arr: Array of index locations in all star arrays for high mass stars
    arr[float] x/y/z_arr: Array of all stars current positions from centre (km)
    arr[float] vx/vy/vz_arr: Array of all stars current velocity components (km/s)
    arr[float] mdot_arr: Array of all stars current mass loss rates (kg/s)
    arr[float] wind_ratio_arr: Array of all stars wind ratio for specific SLR
    arr[float] rdisk_arr: Array of all stars disk radii (typically 100 AU, but I wrote it in a way where it can adjust per star in future) (km)
    float      distance_limit: 
    float      bubble_radius: 
    float      dt: Current timestep (s)
  Outputs:
    arr[float] wind_abs_arr: Array of absorbed wind mass for the particular SLR for every star (kg)
  Once again, if anyone is using this, I am sorry, I really tried not to use numba
  """
  nstars = len(x_arr)
  wind_abs_arr = np.zeros(nstars)
  # Prange doesn't support enumerate, so indexing is manual
  for i in nb.prange(len(lm_id_arr)):
    lm = lm_id_arr[i]
    for j in range(len(hm_id_arr)):
      hm = hm_id_arr[j]
      # Get variables, otherwise contend with a bunch of array math, do it once here
      mdot       = mdot_arr[hm]
      wind_ratio = wind_ratio_arr[hm]
      r_disk     = rdisk_arr[lm]
      # Get star positions
      lm_x , lm_y, lm_z = x_arr[lm], y_arr[lm], z_arr[lm]
      hm_x , hm_y, hm_z = x_arr[hm], y_arr[hm], z_arr[hm]
      # Check to see if using local model and if star is in bubble, if this is the case skip loop iteration
      if distance_limit != 0.0:
        d_sep = ((lm_x - hm_x)**2 + (lm_y - hm_y)**2 + (lm_z - hm_z)**2)**0.5
        if bubble_radius <= d_sep:
          continue
      # Get low mass star velocities
      lm_vx, lm_vy, lm_vz = vx_arr[lm], vy_arr[lm], vz_arr[lm]
      # Calculate disk speed
      disk_spd = (lm_vx**2 + lm_vy**2 + lm_vz**2)**0.5
      d_disk_trav = disk_spd * dt
      # Calculate cross section and wind absorption for global model
      eta_bub  = 0.75 * (r_disk**2) * d_disk_trav / (bubble_radius ** 3)
      wind_abs = wind_ratio * mdot * eta_bub * dt
      # Add result to array
      wind_abs_arr[lm] += wind_abs
  return wind_abs_arr

def evolve_simulation(cluster,converter,gravity,stellar,yields,metadata,t_f,Rc,bar,save,AGB=None):
  """
  Evolve simulation with an adaptive timestep

  Timestep calculation:
  - Timestep is separate from N-body step, which is handled by the numerical solver
  - Timestep is adaptive, and evolves from two main conditions
    - Proximity and sudden intercept velocity with massive stars, timestep is determined by
      the determining the first low-mass star to collide with a massive star assuming it was
      travelling on a collision course at its current velocity, this is then divided by 10
      for good mewasure. This is a fairly fast calculation but can change timestep effectively
      for close encounters
    - If timestep exceeds expected next plot interval, use that instead
  
  Simulation synchronisation:
  - Synchronisation is manual, as there were some issues in getting this to work properly
  - Instead, synchronisation only occurs when it is explicitly needed:
    - Mass
    - Position
    - Supernova kick velocity

  Routines, in order of execution:
  - Determine timestep
  - Check if plotting should occur/simulation should finish
  - Evolve N-body
    - Update positions of stars over tiemstep
    - Resolve supernova kick from previous step
  - Evolve stars
    - Adjust masses, synchronise with N-body for next step
    - Check to see if star has gone supernova, sychronize kick with N-body
  - Evolve disks
    - Check for wind/supernovae injection
    - Calculate mixing fractions, add mass
  - Write data

  Inputs:
  - cluster: contains parameters of each star, handles:
    - Disk parameters, radius, lifetime, whether they are extant
    - Storage of mass injection
    - Supernova kick confirmation
  - gravity: N-body solver, handles:
    - Kinematics
    - Also stores the current canonical time
  - stellar: SeBa stellar evolution solver, which handles:
    - Stellar evolution, of course!
    - Evolutionary phase (MS, PMS, SNRelic, etc.)
    - Mass loss rate
  - yields: Object containing yields, this is serialised and written to disk
  - metadata: 
  - t_f: Final time to evolve to (AMUSE units, Myr)
  - bar: TQDM progress bar
  Outputs:
  - All outputs and inputs are mirrored
  - Finish: a quick bool to determine if the simulation has finished
  - Initially this was just going to recurse, but Python stack depth limitations are strict, so function carries out 
  """

  ### INITIALISATION
  # Get current time
  t = gravity.model_time
  t_start_func = time()
  t_start_init = time()
  # Generate a list of indices for high mass and low mass stars
  hm_id,lm_id = get_high_mass_star_indices(cluster)

  # Get current cluster half-mass radius
  virial_radius = cluster.virial_radius()
  # bar.write("Cluster virial radius = {:.2f}".format(virial_radius.value_in(pc)))
  # half_mass_radius = calc_cluster_half_mass(cluster) | pc
  # bar.write("half_mass_radius = {:.2f}".format(half_mass_radius.value_in(pc)))

  # If all stars have gone supernovae, or gone below high mass threshold, immediately end simulation
  # if len(hm_id) == 0:
  #   bar.write("!!! NO HIGH MASS STARS REMAINING, FINISHING SIMULATION !!!")
  #   finish = True
  #   return cluster,gravity,stellar,yields,metadata,finish,bar
  # Quick sanity check that keys are same this should not happen based on my understanding of AMUSE but it's to be safe
  for i,star in enumerate(cluster):
    if star.key != stellar.particles[i].key or star.key != gravity.particles[i].key:
      raise ValueError("Key mismatch between stellar, cluster and gravity particles, simulation cannot continue!")
  t_fin_init = time()
  # Calculate timestep, previously we used a variable timestep, but it was extremely slow for some reason
  dt = t_f / (n_plot * steps_per_plot)

  # Use adaptive timestep if turned on
  # Calculate the timestep if adaptive timestep enabled
  # if metadata.args.adaptive_timestep:
  #   calc_adaptive_timestep(cluster,hm_id,lm_id,dt,has_interloper=metadata.args.interloper)

  t_new = t + dt

  if metadata.args.verbose:
    bar.write("t = {:.3f} Myr: Finished init, took {:.3f} sec".format(t_new.value_in(myr),t_fin_init - t_start_init))

  # ### TIME-STEP ROUTINES
  # # Get maximal timestep, only used in very sparse environments
  # dt = t_f / n_plot
  # dt_fine = dt / 100 # Fine timestep, for minimising
  # # Calculate safe timestep, based on proximity of stars and their current velocities
  # for i in hm_id:
  #   hm_star = gravity.particles[i]
  #   for j in lm_id:
  #     lm_star = gravity.particles[j]
  #     d = calc_star_distance(hm_star,lm_star)
  #     # If star were on a direct intercept course, calculate time to intercept 
  #     v_int = calc_star_vel(lm_star.vx,lm_star.vy,lm_star.vz)
  #     t_int = d / v_int
  #     # Reduce significantly to find timestep, used for variable step
  #     dt_p = 0.2 * t_int
  #     dt   = min(dt_p,dt)
  #     # Now compare compare to a much smaller timestep to prevent overrefinement and extremely small steps
  #     dt = max(dt,dt_fine)
  # # From minimised dt, find new timestep
  # t_new = t + dt

  # Check timestep is valid against finishing simulation or plotting
  finish = False
  # Check to see if simulation will finish
  if t_new > t_f:
    t_new = t_f
    dt = t_new - t
    finish = True

  ### N-BODY ROUTINES
  # Now, evolve the N-body simulation (most time consuming section of iteration)
  t_start_grav = time()
  # Make a copy of previous position
  gravity_prev = gravity.particles.copy()
  # Evolve model
  gravity.evolve_model(t_new)
  t_fin_grav = time()
  if metadata.args.verbose:
    bar.write("t = {:.3f} Myr: Finished N-body, took {:.3f} sec".format(t_new.value_in(myr),t_fin_grav - t_start_grav))

  ### STELLAR EVOLUTION ROUTINES
  # First, evolve stars according to SEBA model
  t_start_stel = time()
  stellar.evolve_model(t_new)
  t_fin_stel = time()
  if metadata.args.verbose:
    bar.write("t = {:.3f} Myr: Finished SeBa, took {:.3f} sec".format(t_new.value_in(myr),t_fin_stel - t_start_stel))

  ### STELLAR EVOLUTION - SUPERNOVA
  # Add kick velocity to stars gone supernova
  # This has to use array indexing because it requires participation from multiple datasets
  # sn_id = []
  # for i in hm_id:
  #   vx_k = stellar.particles[i].natal_kick_x
  #   if vx_k != 0.0 | kms:
  #     # Check to see if star has been kicked before, if it has, ignore it and move on
  #     if cluster[i].kicked == False:
  #       # Get other properties
  #       vy_k = stellar.particles[i].natal_kick_y
  #       vz_k = stellar.particles[i].natal_kick_z
  #       # Add value to velocities in N-body simulation
  #       gravity.particles[i].vx += vx_k
  #       gravity.particles[i].vy += vy_k
  #       gravity.particles[i].vz += vz_k
  #       # Ensure this is a one time thing by setting SN kick flag to false
  #       cluster[i].kicked = True
  #       # Add index to list of supernovae indices (for later)
  #       sn_id.append(i)

  ### COPYING ROUTINES
  # Now that 
  # Notes: This is the preferred method of exchanging data between codes in AMUSE, timesteps are small enough that synchronous operation does not need to be done, SEBA is much faster than Hermite, therefore there would be minimal gain to do both asynchronously 
  # Channels and pipelines
  stel_to_grav = stellar.particles.new_channel_to(gravity.particles)
  grav_to_clus = gravity.particles.new_channel_to(cluster)
  # Copy over attributes, only mass is copied over to gravity to preserve 0 radius, and doesn't need any other parameters
  stel_to_grav.copy_attributes(["mass"])
  # Cluster is used to store checkpoints and results, there is some overlap between these, which is a waste of memory, but it is fairly minimal
  grav_to_clus.copy()
  
  ### DISK ROUTINES
  ## Calculate SLR deposition for each wind
  # There are probably more reasonable ways to do this, however this is the fastest implementation I could think of
  # Previous version used an entirely python prescription, though this was extremely slower, by around 5 OOM
  # This is passed onto a JIT compiled, nopython function, original python implementation took ~98% of total load
  if len(hm_id) > 0:
    t_start_winds = time()
    # First, split arrays such that they can be read into a NUMBA compliant function (pandas splitting is efficient, accelerated)
    x_arr     = gravity.particles.x.value_in(units.km)
    y_arr     = gravity.particles.y.value_in(units.km)
    z_arr     = gravity.particles.z.value_in(units.km)
    vx_arr    = gravity.particles.vx.value_in(units.km/units.s)
    vy_arr    = gravity.particles.vy.value_in(units.km/units.s)
    vz_arr    = gravity.particles.vz.value_in(units.km/units.s)
    mdot_arr  = - stellar.particles.wind_mass_loss_rate.value_in(units.kg/units.s)
    rdisk_arr = cluster.r_disk.value_in(units.km)
    wind_ratio_26al_arr = cluster.wind_ratio_26al
    wind_ratio_60fe_arr = cluster.wind_ratio_60fe
    # Calculate wind absorption from Global model 
    wind_abs_global_26al = calc_wind_abs(lm_id,hm_id,
                                        x_arr,y_arr,z_arr,
                                        vx_arr,vy_arr,vz_arr,
                                        mdot_arr,
                                        wind_ratio_26al_arr,
                                        rdisk_arr,
                                        distance_limit = 0.0,
                                        bubble_radius = virial_radius.value_in(units.km),
                                        dt = dt.value_in(units.s)) | kg
    wind_abs_global_60fe = calc_wind_abs(lm_id,hm_id,
                                        x_arr,y_arr,z_arr,
                                        vx_arr,vy_arr,vz_arr,
                                        mdot_arr,
                                        wind_ratio_60fe_arr,
                                        rdisk_arr,
                                        distance_limit = 0.0,
                                        bubble_radius = virial_radius.value_in(units.km),
                                        dt = dt.value_in(units.s)) | kg
    # Now calculate wind absorption from Local model
    wind_abs_local_26al = calc_wind_abs(lm_id,hm_id,
                                        x_arr,y_arr,z_arr,
                                        vx_arr,vy_arr,vz_arr,
                                        mdot_arr,
                                        wind_ratio_26al_arr,
                                        rdisk_arr,
                                        distance_limit = r_bub_local_wind.value_in(units.km),
                                        bubble_radius = r_bub_local_wind.value_in(units.km),
                                        dt = dt.value_in(units.s)) | kg
    wind_abs_local_60fe = calc_wind_abs(lm_id,hm_id,
                                        x_arr,y_arr,z_arr,
                                        vx_arr,vy_arr,vz_arr,
                                        mdot_arr,
                                        wind_ratio_60fe_arr,
                                        rdisk_arr,
                                        distance_limit = r_bub_local_wind.value_in(units.km),
                                        bubble_radius = r_bub_local_wind.value_in(units.km),
                                        dt = dt.value_in(units.s)) | kg
    # Finish up! Copy values back into original wind arrays, also fast, because this is accelerated
    cluster.mass_26al_global += wind_abs_global_26al 
    cluster.mass_60fe_global += wind_abs_global_60fe
    cluster.mass_26al_local  += wind_abs_local_26al
    cluster.mass_60fe_local  += wind_abs_local_60fe
    t_fin_winds = time() 
    if metadata.args.verbose:
      bar.write("t = {:.3f} Myr: Finished winds, took {:.3f} sec".format(t_new.value_in(myr),t_fin_winds - t_start_winds))

  ### SUPERNOVA ROUTINES
  # This one doesn't have to be written for numba, since it will only happen every time a supernova occurs, so a handful of times per simulation
  if len(hm_id) > 0:
    for i in hm_id:
      mdot = - stellar.particles[i].wind_mass_loss_rate
      if mdot == 0.0 | msolyr:
        if cluster[i].kicked == False:
          # Alert user, a supernovae is a big event afterall! Think of all the bright eyed particle physics postgrads stuck down a mineshaft at Super Kamiokande! They live to see that kind of thing!
          bar.write("Star #{} has gone supernova!".format(i))
          # Get supernoave yield for star
          al26_sn = cluster[i].sn_yield_26al
          fe60_sn = cluster[i].sn_yield_60fe
          # Now check each star to deposit SLRs
          for j in lm_id:
            d = calc_star_distance(gravity.particles[i],gravity.particles[j])
            # if d < r_bub_local_sne: # Used to be that there was a bubble size for SNe, there isn't anymore
            r_disk   = cluster[j].r_disk
            eta_disk = calc_eta_disk_sne(r_disk,d)
            al26_inj = al26_sn * eta_disk
            fe60_inj = fe60_sn * eta_disk
            # Store as appropriate
            cluster[j].mass_26al_sne += al26_inj
            cluster[j].mass_60fe_sne += fe60_inj
          # Enable kick flag, to ensure this doesn't repeat in the next iteration
          cluster[i].kicked = True

  ### INTERLOPER ROUTINES
  # Handle injection of SLRs from interloper
  t_start_interloper = time()
  if metadata.args.interloper:
    # Quick check to find the interloper, it __should__ be the last item in the list, if it isn't, search for it
    if cluster[-1].is_interloper == True: interloper = cluster[-1]
    else:
      interlopers = cluster[cluster.is_interloper == True]
      if len(interlopers) > 1:
        raise ValueError("More than 1 interloper found! That's not supposed to happen!")
      else:
        interloper = interlopers[0]
    int_x,int_y,int_z = interloper.x,interloper.y,interloper.z
    time_offset = metadata.args.interloper_offset_time
    interloper_bubble_radius = metadata.args.interloper_bubble_radius
    interloper_time = t - time_offset
    if interloper_time > 0.0 | myr:
      # bar.write("Interloper running this step!")
      interloper_26al_rate = AGB.interp_value("26al_mass_loss_rate",interloper_time)
      interloper_60fe_rate = AGB.interp_value("60fe_mass_loss_rate",interloper_time)
      if interloper_26al_rate > 0.0 | msolyr or interloper_60fe_rate > 0.0 | msolyr:
        for i in lm_id:
          if cluster[i].is_interloper == False:
            # if cluster[i].disk_alive == True:
            lm_x,lm_y,lm_z    = cluster[i].x,cluster[i].y,cluster[i].z
            # lm_vx,lm_vy,lm_vz = cluster[i].vx,cluster[i].vy,cluster[i].vz
            r_disk = cluster[i].r_disk
            int_x_o,int_y_o,int_z_o = gravity_prev[-1].x,gravity_prev[-1].y,gravity_prev[-1].z
            lm_x_o,lm_y_o,lm_z_o    = gravity_prev[i].x,gravity_prev[i].y,gravity_prev[i].z

            # Calculate the fraction of the timestep that the interloper and disk pass within 0.1 pc of each other
            # This is used as this can account for near misses, the fraction can then be used to derive a more accurate path distance
            intersection_frac = calc_intersection(int_x_o.value_in(pc),
                                                  int_y_o.value_in(pc),
                                                  int_z_o.value_in(pc),
                                                  int_x.value_in(pc),
                                                  int_y.value_in(pc),
                                                  int_z.value_in(pc),
                                                  lm_x_o.value_in(pc),
                                                  lm_y_o.value_in(pc),
                                                  lm_z_o.value_in(pc),
                                                  lm_x.value_in(pc),
                                                  lm_y.value_in(pc),
                                                  lm_z.value_in(pc),
                                                  0.1)

            if intersection_frac != 0.0:
              interlopers = cluster[cluster.is_interloper == True]
              # Calculate disk speed
              # disk_spd = (lm_vx**2 + lm_vy**2 + lm_vz**2)**0.5
              # d_disk_trav = disk_spd * dt
              d_disk_trav = ((lm_x - lm_x_o)**2 + (lm_y - lm_y_o)**2 + (lm_z - lm_z_o)**2)**0.5
              d_disk_trav *= intersection_frac
              eta_bub  = 0.75 * (r_disk**2) * d_disk_trav / (interloper_bubble_radius ** 3)
              inter_abs_26al = interloper_26al_rate * eta_bub * dt
              inter_abs_60fe = interloper_60fe_rate * eta_bub * dt
              cluster[i].mass_26al_agb += inter_abs_26al
              cluster[i].mass_60fe_agb += inter_abs_60fe
              cluster[i].mass_26al_agb_raw += inter_abs_26al
              cluster[i].mass_60fe_agb_raw += inter_abs_60fe
  
      if args.interloper_trajectory:
        with open("interloper_trajectory.dat","a") as traj_file:
          traj_t_sim = t.value_in(myr)
          traj_t_int = interloper_time.value_in(myr)
          traj_x,traj_y,traj_z = int_x.value_in(pc),int_y.value_in(pc),int_y.value_in(pc)
          bary = cluster.center_of_mass()
          bary_dist = (((int_x-bary[0])**2 + (int_y-bary[1])**2 + (int_z-bary[2])**2)**0.5).value_in(pc)
          traj_file.write("{:.3e},{:.3e},{:.3e},{:.3e},{:.3e},{:.3e}\n".format(traj_t_sim,traj_t_int,traj_x,traj_y,traj_z,bary_dist))
      
  
  t_fin_interloper = time()
  if metadata.args.verbose:
    bar.write("t = {:.3f} Myr: Finished interloper, took {:.3f} sec".format(t_new.value_in(myr),
                                                                            t_fin_interloper - t_start_interloper))

  ### DECAY ROUTINES
  # Progressively remove SLRs from disks through radioactive decay
  t_start_decay = time()
  half_life_26al = 0.717 | myr
  half_life_60fe = 2.600 | myr
  decay_frac_26al = np.exp((-dt*0.693147)/half_life_26al)
  decay_frac_60fe = np.exp((-dt*0.693147)/half_life_60fe)
  # Perform decay
  # For 26Al variables
  cluster.mass_26al_local  *= decay_frac_26al
  cluster.mass_26al_global *= decay_frac_26al
  cluster.mass_26al_sne    *= decay_frac_26al
  # For 60Fe variables
  cluster.mass_60fe_local  *= decay_frac_60fe
  cluster.mass_60fe_global *= decay_frac_60fe
  cluster.mass_60fe_sne    *= decay_frac_60fe
  # For interloper
  if metadata.args.interloper:
    cluster.mass_26al_agb    *= decay_frac_26al
    cluster.mass_60fe_agb    *= decay_frac_60fe

  t_fin_decay = time()
  if metadata.args.verbose:
    bar.write("t = {:.3f} Myr: Finished decay, took {:.3f} sec".format(t_new.value_in(myr),t_fin_decay - t_start_decay))

  ### CONDENSE ROUTINES
  for i in lm_id:
    if cluster[i].disk_alive == True:
      if cluster[i].tau_disk >= t_new:
        cluster[i].mass_26al_local_final  = cluster[i].mass_26al_local
        cluster[i].mass_26al_global_final = cluster[i].mass_26al_global
        cluster[i].mass_26al_sne_final    = cluster[i].mass_26al_sne
        cluster[i].mass_60fe_local_final  = cluster[i].mass_60fe_local
        cluster[i].mass_60fe_global_final = cluster[i].mass_60fe_global
        cluster[i].mass_60fe_sne_final    = cluster[i].mass_60fe_sne
        if metadata.args.interloper:
          cluster[i].mass_26al_agb_final = cluster[i].mass_26al_agb
          cluster[i].mass_60fe_agb_final = cluster[i].mass_60fe_agb
      if cluster[i].tau_disk < t_new:
        # if metadata.args.verbose:
        bar.write("Disk of low-mass star #{} has condensed".format(i))
        cluster[i].disk_alive = False
  
  ### HOUSEKEEPING ROUTINES
  # Update progress bar using dt value
  # This is a fairly brute force way of doing it, but the progress bar has always been a bit
  # Inaccurate when it comes to indeterminate numbers of steps
  prog_old = bar.n
  prog_new = t_new.value_in(myr)
  prog_del = prog_new - prog_old
  bar.update(prog_del)

  if save == True:
    # Update the conditions of metadata object
    metadata.update(t_new)
    # Update yields
    yields.update_state(gravity.model_time,cluster)
    # Save data to checkpoint
    filename = metadata.filename
    nfile = metadata.most_recent_checkpoint
    save_checkpoint(filename,nfile,cluster,converter,yields,metadata,bar=bar)

  t_fin_func = time()
  if metadata.args.verbose:
    bar.write("! Finished step t = {:.3f} Myr, took {:.3f} sec".format(t_new.value_in(myr),t_fin_func - t_start_func))

  # the main function
  # Finish simulation and return, if finish = False, then this will be run again inside a loop in
  return cluster,gravity,stellar,yields,metadata,finish,bar


@nb.njit(parallel=True)
def calc_min_intercept_time(x,y,z,vx,vy,vz,lm_id,hm_id):
  nlm,nhm = len(lm_id),len(hm_id)
  int2_arr = np.zeros(nlm*nhm) # Array to store intercept time results^2 (seconds^2)
  spd2_arr = np.zeros(nhm)     # Array to store interception candidate speeds^2 (km^2/s^2)
  for i in range(nhm):
    hm = hm_id[i]
    vx1,vy1,vz1 = vx[hm],vy[hm],vz[hm] # Get values of velocities
    spd2 = vx1**2 + vy1**2 + vz1**2 # Calculate
    print(i,spd2**0.5)
    spd2_arr[i] = spd2
  for i in nb.prange(nlm):
    lm = lm_id[i]
    for j in range(nhm):
      hm = hm_id[j]
      ij = i + (j*nlm)
      x1,y1,z1 = x[lm],y[lm],z[lm]
      x2,y2,z2 = x[hm],y[hm],z[hm]
      spd2 = spd2_arr[j]
      d2 = (x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2
      int2_arr[ij] = d2 / spd2
  int_t_min = np.min(int2_arr) ** 0.5
  return int_t_min
def calc_adaptive_timestep(cluster,hm_id,lm_id,dt,has_interloper = False):
  if has_interloper:
    interloper_id = -1
    hm_id.append(interloper_id)
  x  = cluster.x.value_in(units.km)
  y  = cluster.y.value_in(units.km)
  z  = cluster.z.value_in(units.km)
  vx = cluster.vx.value_in(units.kms)
  vy = cluster.vy.value_in(units.kms)
  vz = cluster.vz.value_in(units.kms)
  int_t_min  = calc_min_intercept_time(x,y,z,vx,vy,vz,lm_id,hm_id) | units.s
  print(dt.value_in(myr))
  print(int_t_min.value_in(myr))
  n_substeps = ceil((dt / int_t_min) * 1)
  print(n_substeps)
  sys.exit()

def calc_intersection(x1o,y1o,z1o,
                      x1n,y1n,z1n,
                      x2o,y2o,z2o,
                      x2n,y2n,z2n,
                      r,n=1024):
  """
  Calculate the fraction of a step that two particles spend a certain distance apart
  Assumes that both particles are travelling in straight lines
  There are better ways to do this, I'm sure, but it's relatively fast and I am tired
  Inputs:
  - x1o,y1o,z1o: initial "old" positions of particle 1
  - x1n,y1n,z1n: final "new" positions of particle 1
  - x2o,y2o,z2o: initial "old" positions of particle 2
  - x2n,y2n,z2n: final "new" positions of particle 2
  - r: separation distance
  - n: number of division substeps
  Outputs:
  - nf: step proximity fraction, as described above
  """

  x1i = np.linspace(x1o,x1n,n)
  y1i = np.linspace(y1o,y1n,n)
  z1i = np.linspace(z1o,z1n,n)
  x2i = np.linspace(x2o,x2n,n)
  y2i = np.linspace(y2o,y2n,n)
  z2i = np.linspace(z2o,z2n,n)


  # sys.exit()

  # Do the calculation
  ri = ((x1i-x2i)**2 + (y1i-y2i)**2 + (z1i-z2i)**2) ** 0.5
  ni = np.array(ri <= r).sum()
  nf = ni / n
  return nf

  

def get_high_mass_star_indices(cluster):
  """
  Rather than creating subsets of data, which use a lot of memory and have synchronisation issues
  we instead determine which stars are massive or not, and returns the index, such that all 
  simulation objects can be addressed at once.
  While this is run at every step, it is a comparatively lightweight loop, and scales such that
  O(n)

  Inputs:
    - cluster: The cluster particle array, which has the masses for each star
  Outputs:
    - hm_id: an array of high mass stars indices (high mass defined as >= 13 Msol)
    - lm_id: an array of low mass star indices (low mass defined as <= 3 Msol)
  """
  hm_id = []
  lm_id = []
  for i,star in enumerate(cluster):
    if star.mass >= 13.0 | msol:
      hm_id.append(i)
    if star.mass >= 0.1 | msol and star.mass <= 3.0 | msol:
      # if star.disk_alive == True: # TEST PATCH
      lm_id.append(i)
  return hm_id,lm_id

def disk_lifetime():
  """
  Calcualting a disk lifetime, we assume a mean disk lifetime of 5Myr, with an exponential 
  distribution to calculate the decay time.
  Calculating the decay time ahead of the simulation - effectively predetermining the fate of the disk - has a number of dire philosophical connotations, but is probably fine for an N-body simulation.

  This is based off of estimated lifetimes from:
    Richert, A. J. W., Getman, K. V., Feigelson, E. D., Kuhn, M. A., Broos, P. S., Povich, M. S.,
    Bate, M. R., & Garmire, G. P. (2018).
    Circumstellar disk lifetimes in numerous galactic young stellar clusters.
    Monthly Notices of the Royal Astronomical Society, 477, 5191–5206.
    https://doi.org/10.1093/mnras/sty949 
  """
  
  # Quick modification to this, see Circumstellar disc lifetimes in numerous galactic young stellar clusters, Richert et al. 2018
  lam = 2.885 # Scale height, mean lifetime of 2.885 myr, corresponds to t1/2 of 2myrs 

  tau = exponential(lam) | myr
  return tau

def calc_v(vx,vy,vz):
  return ((vx*vx) + (vy*vy) + (vz*vz))**0.5

def calc_eta_bubble_wind(r_disk,r_bub,d_bub_trav):
  """
  Calculate the cross section of sweeped-up wind from a disk.
  As a disk traverses through a wind blown bubble, mass from the stellar wind is dredged up, thus
  deposition rate is a factor of the size of the bubble, the velocity between the stars and the
  size of the disk.
  This assumes that the mass loss rate of the wind blowing the bubble tapers off drastically
  at the edge of the bubble.
  The al26 yield can be calculated by multiplying the resultant value by the mass of al26 emitted
  by the star
  """
  # Use this value to calculate the total wind efficiency
  eta_bub = 0.75 * (r_disk**2) * d_bub_trav / (r_bub**3)
  return eta_bub
  
def calc_star_vel(vx,vy,vz):
  """
  Calculate star scalar velocity from individual velocity components
  Whilst I try and keep most of my code well-documented, there is not much else I can write about
  this - Joe

  Inputs:
    vx: Star x-component velocity
    vy: Star y-component velocity
    vz: Star z-component velocity
  Outputs:
    v_star: Star scalar velocity
  """
  # I really shouldn't have to document this
  v_star = np.sqrt(vx**2 + vy**2 + vz**2)
  return v_star

def calc_star_rel_vel(vx1,vy1,vz1,vx2,vy2,vz2):
  """
  Calculate the relative velocity of two stars in cartesian space
  Again, I'm not sure what to add here - Joe

  Inputs:
    vx1,vy1,vz1: Primary star x,y,z velocity components
    vx2,vy2,vz2: Secondary star x,y,z velocity components
  Outputs:
    rel_vel: Relative scalar velocity between stars
  """
  # Calculate the relative components
  vrx,vry,vrz = (vx1-vx2),(vy1-vy2),(vz1-vz2)
  # Square it, mutiplying is usually faster
  vrx2,vry2,vrz2 = (vrx*vrx),(vry*vry),(vrz*vrz)
  rel_vel = np.sqrt(vrx2 + vry2 + vrz2)
  return rel_vel

def calc_eta_disk_sne(r,d):
  """
  Calculates the proportion of stellar wind adsorbed onto the stellar wind which condenses

  The proprtion is calculated from three terms:
    - eta_inj: Injection efficiency, proportion of wind that successfully interacts with the disk
    - eta_cond: Condensation efficiency, proportion of wind that condenses onto the disk 
    - eta_geom: Geometric efficiency, based on disk radius, r, disk distance to donor star, d, and
      disk angle(theta)
      - This is given by the formula:
        (pi * r^2) / (4 * pi * d^2) * cos(theta)
      - In the case of these simulations, the disks are at a constant angle of 60 degrees, to reduce
        complexity
  Furthermore, this assumes a grain size of 0.1 microns, which derives the values of eta_cond and
  eta_inj.
  The final proportion, eta_total is calculated by combining these values:
  eta_total = eta_cond * eta_inj * eta_geom

  This is based off of calculations and values in section 2.2 of the paper:
    Lichtenberg, T., Parker, R. J., & Meyer, M. R. (2016).
    Isotopic enrichment of forming planetary systems from supernova pollution.
    Monthly Notices of the Royal Astronomical Society, 462, 3979–3992.
    https://doi.org/10.1093/mnras/stw1929
  
  Inputs:
    - r: Disk radius
    - d: Disk distance to donor star
    > These values can be in any units as long as they are:
      - The same unit
      OR
      - If they are stored as AMUSE units, which will allow for conversion
  Outputs:
    - eta_total: Total efficiency of adsorption
  """

  # Constants
  cos60    = 0.5 # Hard-coding cos(60)
  eta_cond = 0.5 # Condensation efficiency
  eta_inj  = 0.7 # Injection efficiency
  # Now calculate eta_geom
  eta_geom = (cos60 * r**2) / (4 * d**2)
  # Calculate the whole thing
  eta_total = eta_cond * eta_inj * eta_geom
  return eta_total

@nb.njit()
def half_mass_calc_numba(d2_index_sort,d2,masses,cluster_mhalf):
  mass_counter = 0.0
  for i in d2_index_sort:
    mass = masses[i]
    mass_counter += mass
    if mass_counter >= cluster_mhalf:
      half_mass_radius = d2[i]**0.5
      break
  return half_mass_radius
def calc_cluster_half_mass(cluster):
  """
  Calculate half mass radius of a cluster in parsecs (returned as float, not amuse units)
  """
  from amuse.datamodel import Particles
  # First, find the barycentre, as this may not necessarily be (0,0,0)
  bary = cluster.center_of_mass()
  masses = cluster.mass.value_in(msol)
  # Now calculate the half mass of the cluster
  cluster_mhalf = (cluster.mass.sum()/2.).value_in(msol)
  # Create a test particle at the barycentre for d2 calculations
  test = Particles(1)
  test.x,test.y,test.z = [bary[0]],[bary[1]],[bary[2]]
  # Calculate d2 from AMUSE built in function
  d2 = cluster.distances_squared(test)[:,0].value_in(pc*pc)
  d2_index_sort = d2.argsort()
  half_mass_radius = half_mass_calc_numba(d2_index_sort,d2,masses,cluster_mhalf)
  return half_mass_radius

def calc_star_distance(star1,star2):
  """
  Calculates distance between star 1 and star 2, uses amuse values so unit is flexible.
  You should know this calculation.
  """
  x1,y1,z1 = star1.x,star1.y,star1.z
  x2,y2,z2 = star2.x,star2.y,star2.z
  d = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
  return d

@nb.njit
def maschberger(m,G_lower,G_upper):
  """
  Calculate Maschberger probability function for given auxiliary functions and masses
  """
  mu = 0.2 # Average star mass
  a  = 2.3 # Low mass component
  b  = 1.4 # High mass component
  A = (((1-a) * (1-b))/mu) * (1/(G_upper-G_lower))
  p = A * ((m / mu)**(-a)) * ((1 + (m/mu)**(1-a))**(-b))
  return p

def maschberger_aux(m):
  """
  Auxiliary function for Maschberger distribution for a given mass
  """
  mu = 0.2 # Average star mass
  a  = 2.3 # Low mass component
  b  = 1.4 # High mass component
  return (1 + ((m/mu)**(1-a)))**(1-b)

@nb.njit
def gen_mass_numba(G_lower,G_upper,pm_lower,pm_upper,min_mass,max_mass,nstars):
  """
  Generate an array of star masses along the Maschberger distribution
  Setup done in generate_masses()
  """
  masses = np.zeros(nstars)
  i = 0
  while i < nstars:
    m = np.random.uniform(min_mass,max_mass)
    pm = np.random.uniform(pm_lower,pm_upper)
    if pm < maschberger(m,G_lower,G_upper):
      masses[i] = m
      i += 1
  return masses

def generate_masses(nstars,
                    min_mass=0.01,max_mass=150.,m_lower=0.01,m_upper=150.,
                    no_massive_star_requirement=False):
  """
  Generate a series of masses for stars using the Maschberger IMF, ensures that at least 1 massive star is included
  """

  G_lower  = maschberger_aux(m_lower)
  G_upper  = maschberger_aux(m_upper)
  pm_upper = maschberger(m_lower,G_lower,G_upper)
  pm_lower = maschberger(m_upper,G_lower,G_upper)
  masses   = []

  print("Sampling masses... ",end="")
  masses = gen_mass_numba(G_lower,G_upper,pm_lower,pm_upper,min_mass,max_mass,nstars)
  if no_massive_star_requirement == False:
    if max(masses) < 13.0:
      print("No massive stars in cluster! Re-rolling!")
      masses = generate_masses(nstars,
                               min_mass=min_mass,
                               max_mass=max_mass,
                               m_lower=m_lower,
                               m_upper=m_upper,
                               no_massive_star_requirement=no_massive_star_requirement)

  ii,iii = 0,0
  for i,mass in enumerate(masses):
    if mass > 13.0:
      ii += 1
    if mass >= 0.1 and mass <= 3.0:
      iii += 1
  print("Finished! {} massive stars and {} disk-bearing stars generated".format(ii,iii))
  # Convert list into array
  masses = np.asarray(masses)
  return masses

def spawn_interloper(cluster,interloper_mass,interloper_distance,closest_approach_radius,interloper_velocity,time_offset,Rc):
  """
  Add an additional, evolved AGB star to the simulation on a collision course with the star forming
  region.

  Interloper is placed at the position (-2rc,ro,0) in the simulation, where rc is the star forming region radius and ro is the offset radius

  This function adds the following properties to the cluster system:
  - is_interloper: Boolean for whether a star is the interloper or not
  The remainder of the handling for the interloper is governed separately
  
  Inputs:
    - cluster: Simulation cluster system
    - stellar: Stellar evolution system
    - gravity: N-body system
    - interloper_mass: Mass of AGB star (Msol)
    - closest_approach: Closest approach distance of interloper (pc)

  Outputs:
    - cluster: Input cluster system with interloper appended
    - stellar: Input stellar evolution system with interloper appended
    - gravity: Input N-body system with interloper appended

  References:
    Parker, R. J., & Schoettler, C. (2023). Isotopic Enrichment of Planetary Systems from Asymptotic Giant Branch Stars. The Astrophysical Journal Letters, 952(1), L16. https://doi.org/10.3847/2041-8213/ace24a
  """
  converter  = nbody_system.nbody_to_si(Rc,interloper_mass)
  interloper = plummer.new_plummer_model(1, converter)
  # Write mass
  interloper[0].mass = interloper_mass
  # Write positions
  interloper[0].x = interloper_distance
  interloper[0].y = closest_approach_radius 
  interloper[0].z = 0.0 | pc
  # Write velocities
  interloper[0].vx = interloper_velocity
  interloper[0].vy = 0.0 | kms
  interloper[0].vz = 0.0 | kms
  interloper[0].is_interloper = True
  interloper[0].age = 1 | myr
  cluster.add_particles(interloper)
  # print(cluster)
  return cluster

def init_cluster(model, nstars, Rc, SLRs,
                 min_mass,max_mass,
                 no_massive_star_requirement=False,
                 r_disk=100.):
  """
  Initialise cluster
  Input:
    - model:  Cluster model to use
    - nstars: Number of stars
    - Rc:     Cluster radius in parsecs
    - nmass:  Override, number of massive stars (M* > 12Msol) to include
  Output:
    - cluster: An array containing a series of stars
  """

  # First, generate a series of masses
  masses = generate_masses(nstars,
                           no_massive_star_requirement=no_massive_star_requirement,
                           min_mass=min_mass,max_mass=max_mass)
  # Set basic simulation parameters
  Mcluster  = sum(masses) | msol
  # Now create the cluster
  print("Generating cluster...")
  masses    = masses | msol # Convert to AMUSE units
  converter = nbody_system.nbody_to_si(Rc,Mcluster) # Create nbody units converter
  # Build the cluster based on a plummer model
  # TODO add more models
  if model == "plummer":
    cluster = plummer.new_plummer_model(nstars, converter)
  elif model == "fractal":
    seed = np.random.seed()
    cluster = new_fractal_cluster_model(N = nstars,
                                        fractal_dimension = args.fractal_dimension,
                                        random_seed = seed,
                                        convert_nbody = converter) 
  else:
    raise ValueError("Invalid choice of cluster model, must be either \"plummer\" or \"fractal\"!")
  # Add generated masses to cluster
  cluster.mass = masses

  # Add parameters to cluster as a whole
  # Unfortunately as a lot of these parameters get passed to the stellar evolution and clustering systems you can't add these as classes, which is a bit of a shame really
  # Additionally you can't delete class parameters in a particle, though I'm not sure why you can't meaning it can't be subset upon save
  # Separating them out is also an issue, as it tremendously complicates the checkpointing system, while also rendering pickle impossible to implement
  # As such each value is now its own parameter, which does complicate things!

  # Step 1: Calculate star properties that are mass dependent, these can't be vectorised as particles

  for star in cluster:
    # Star properties
    star.radius = 0. | au # Stars are dimensionless
    star.kicked = False # Supernova kick flag
    # Protoplanetary disk properties
    star.m_disk_gas  = 0.1 * star.mass
    star.m_disk_dust = 0.01 * star.m_disk_gas
    star.r_disk      = r_disk | au # Stars have a common initial disk size
    star.tau_disk    = disk_lifetime() # Calculate disk lifetime from random sample distribution
    # SLR properties
    ## Aluminium group
    # NB: stable isotope masses calculated from star mass not dust mass, this produces the same value
    #     but looks different to how presented in my paper and all other papers, specifically 
    #     Parker et al. 2023, I've tripped myself up like ~twice~ thrice now thinking that I made a typo here
    #     again, it is the _same value, just represented differently_
    star.mass_27al = 8.500e-6 * star.mass # Calculate stable Al mass from Chrondritic samples
    star.mass_26al_local   = 0.0 | kg
    star.mass_26al_global  = 0.0 | kg
    star.mass_26al_sne     = 0.0 | kg
    star.mass_26al_agb     = 0.0 | kg
    star.mass_26al_agb_raw = 0.0 | kg # Special raw value where no disk progression or radio decay is done
    # Final masses of aluminium
    star.mass_26al_local_final  = 0.0 | kg
    star.mass_26al_global_final = 0.0 | kg
    star.mass_26al_sne_final    = 0.0 | kg
    star.mass_26al_agb_final    = 0.0 | kg
    ## Iron group
    star.mass_56fe = 1.828e-4 * star.mass # Claculate stable Fe mass from Chrondritic samples
    star.mass_60fe_local   = 0.0 | kg
    star.mass_60fe_global  = 0.0 | kg
    star.mass_60fe_sne     = 0.0 | kg
    star.mass_60fe_agb     = 0.0 | kg
    star.mass_60fe_agb_raw = 0.0 | kg # Special raw value where no disk progression or radio decay is done
    # Final masses of iron
    star.mass_60fe_local_final  = 0.0 | kg
    star.mass_60fe_global_final = 0.0 | kg
    star.mass_60fe_sne_final    = 0.0 | kg
    star.mass_60fe_agb_final    = 0.0 | kg
    # Mass dependent based on bracketing (if a star is massive or not)
    ## High mass group, stars greater than 13 solar masses (minimum value in Limongi & Cheiffi)

    if star.mass >= 13.0 | msol:
      star.disk_alive = False # Massive stellar disk not simulated
      star.total_wind_loss = calc_total_mass_loss(star.mass)
      # Wind yields for each SLR
      ## 26Al
      star.wind_yield_26al = calc_slr_yield(star.mass,
                                            SLRs["Al26"].wind_mass,
                                            SLRs["Al26"].wind_yield)
      star.wind_ratio_26al = calc_wind_ratio(star.total_wind_loss,star.wind_yield_26al)
      ## 60Fe
      star.wind_yield_60fe = calc_slr_yield(star.mass,
                                            SLRs["Fe60"].wind_mass,
                                            SLRs["Fe60"].wind_yield)
      star.wind_ratio_60fe = calc_wind_ratio(star.total_wind_loss,star.wind_yield_60fe)
      # Supernova yields for each SLR
      star.sn_yield_26al = calc_slr_yield(star.mass,
                                          SLRs["Al26"].sne_mass,
                                          SLRs["Al26"].sne_yield) # 26Al yield
      star.sn_yield_60fe = calc_slr_yield(star.mass,
                                          SLRs["Fe60"].sne_mass,
                                          SLRs["Fe60"].sne_yield) # 60Fe yield
    if star.mass >= 0.1 | msol and star.mass <= 3.0 | msol:
      star.disk_alive = True
      star.total_wind_loss = 0.0 | msol # Mass loss for low mass stellar winds can be neglected


  # Finish up and return cluster
  print("Cluster done!")
  # print(cluster)
  return cluster,converter

def main(args):
  """
  main()
  
  This function builds the simulation by calling other functions, in particular:
  - Initialises star cluster
  - Initialises N-body and stellar evolution
  - Starts the progress bar (very important)
  - Writes header data to output files
  - Runs the simulation in a continuous loop
  - Runs the simulation in a continuous loop
  - Runs th-you get it 
  """

  # Initialise dictionary containing properties of short-lived radioisotopes
  SLRs = read_SLRs(module_directory+"/slr-abundances.csv")

  if args.n == None or args.rc == None:
    if args.reload == "":
      raise Exception("Input arguments need to either be loading a checkpoint or defining a simulation")
    else:
      pass
    
  # Begin building the cluster
  nstars = args.n # Number of stars
  Rc     = args.rc | pc  # Plummer sphere half-mass-radius
  t_f    = args.final_time | myr # Final simulation time in Myr
  # Initialise star cluster and star masses, or load from disk

  if args.reload == "":
    cluster,converter = init_cluster("plummer",nstars,Rc,SLRs,
                                     args.star_min_mass,args.star_max_mass,
                                     no_massive_star_requirement=args.no_massive_star_requirement,
                                     r_disk = args.disk_radius)
    used_checkpoint = False
  else:
    print("! Loading from file {}...".format(args.reload))
    # First, check
    if args.n_checkpoint == None:
      checkpoint_number = most_recent_checkpoint(args.reload)
    else:
      checkpoint_number = args.n_checkpoint
    cluster,converter,yields,metadata = load_checkpoint(args.reload,checkpoint_number)
    metadata.update_access_time() # Inform metadata has been accessed
    used_checkpoint = True

  AGB = None
  if args.interloper:
    if args.reload == "":
      # Convert interloper mass
      args.interloper_mass = args.interloper_mass | msol
      # Convert interloper bubble size
      args.interloper_bubble_radius = args.interloper_bubble_radius | pc
      # Convert interloper offset radius
      if args.interloper_radius == None:
        args.interloper_radius = uniform(0.0,args.rc)
      args.interloper_radius = args.interloper_radius | pc
      # Convert interloper offset radius
      if args.interloper_distance == None:
        args.interloper_distance = 2 * args.rc
      args.interloper_distance = - args.interloper_distance | pc
      # Convert interloper velocity
      if args.interloper_velocity == None:
        args.interloper_velocity = uniform(0.0,100.0)
      args.interloper_velocity = args.interloper_velocity | kms
      # Manage offset time
      args.interloper_offset_time = args.interloper_offset_time | myr
      # Add interloper to cluster
      print("!!! Spawning an interloper with properties: !!!")
      print("  > Mass:             {:.1f} Msol".format(args.interloper_mass.value_in(msol)))
      print("  > Bubble radius:    {:.3f} pc".format(args.interloper_bubble_radius.value_in(pc)))
      print("  > Initial dist.:    {:.3f} pc".format(args.interloper_distance.value_in(pc)))
      print("  > Close Approach:   {:.3f} pc".format(args.interloper_radius.value_in(pc)))
      print("  > Velocity:         {:.3f} km/s".format(args.interloper_velocity.value_in(kms)))
      print("  > Offset:           {:.3f} Myr".format(args.interloper_offset_time.value_in(myr)))
      print("  > Write trajectory: {}".format(args.interloper_trajectory))

      # Read AGBs, bad code incoming
      AGBs = read_AGBs()
      for AGB_candidate in AGBs:
        if args.interloper_mass == AGB_candidate.mass:
          AGB = AGB_candidate
      if AGB == None:
        validmasses = []
        for AGB in AGBs:
          validmasses.append(AGB.mass.value_in(msol))
        raise ValueError("NO VALID INTERLOPER MASS, MUST BE {} MSOL".format(validmasses))
      # Finishing checks and balances, adding an interloper to the cluster
      cluster = spawn_interloper(cluster,
                                 args.interloper_mass,
                                 args.interloper_distance,
                                 args.interloper_radius,
                                 args.interloper_velocity,
                                 args.interloper_offset_time,
                                 Rc)

  # Initialise N-Body simulation
  if gravity_model == "hermite":
    from amuse.lab import Hermite
    gravity = Hermite(converter,number_of_workers=workers)
  elif gravity_model == "bhtree":
    from amuse.community.bhtree.interface import BHTree
    gravity = BHTree(converter,number_of_workers=workers)
  elif gravity_model == "ph4":
    from amuse.community.ph4.interface import ph4
    gravity = ph4(converter,number_of_workers=workers)
  elif gravity_model == "hermite0":
    from amuse.community.hermite0.interface import Hermite
    gravity = Hermite(converter,number_of_workers=workers,mode="GPU")
  else:
    raise ValueError("Invalid N body solver selection!")
  
  # print(cluster[-1])
  # # print(cluster)
  # sys.exit()

  gravity.particles.add_particles(cluster)
  # Initialise stellar evolution simulation
  stellar = stellar_model()
  stellar.particles.add_particles(cluster)

  # Quickly change model times to simulation time, otherwise time reset every reload
  if used_checkpoint == True:
    # Change simulation times
    gravity.model_time = metadata.time
    stellar.model_time = metadata.time

  if used_checkpoint == False:
    # Generate the rest of the simulation data
    metadata = Metadata(args,t_f) # Metadata object
    yields   = Yields(metadata.filename) # Yields
    yields.update_state(gravity.model_time,cluster) # Perform first flush of data
    # Now save all new files so original state of simulation can be restored
    save_checkpoint(metadata.filename,0,cluster,converter,yields,metadata)

  # Initialise progress bar
  bar = tqdm(total = t_f.value_in(myr),desc="Simulation",position=0,unit="Myr")
  # Begin simulation, open ended until finish is true
  finish = False
  save   = False
  n_iter = 0

  while finish == False:
    if n_iter % steps_per_plot == 0:
      save = True
    else:
      save = False
    cluster,gravity,stellar,yields,metadata,finish,bar = evolve_simulation(cluster,converter,gravity,stellar,yields,metadata,t_f,Rc,bar,save,AGB=AGB)
    n_iter += 1

  gravity.stop()
  stellar.stop()
  bar.close()
  print("!!! Finished !!!")
  return

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Calculate orbital trajectories and Al26 enrichment of a stellar cluster")
  parser.add_argument("-n", default=None, type=int, help="Number of stars in cluster")
  parser.add_argument("-rc", default=None, type=float, help="Cluster radius (pc)")
  parser.add_argument("-r","--reload", type=str, default="", help="Base name of files to RELOAD")
  parser.add_argument("-nc", "--n_checkpoint",type=int,default=None,help="Which checkpoint file to load, defaults to highest number")
  parser.add_argument("-m", "--model", type=str, default="plummer", help="Which model to use, defaults to Plummer sphere, can also use fractal model")
  parser.add_argument("-d","--fractal_dimension",type=float,default=2.0,help="Dimension parameter for fractal model")
  parser.add_argument("-rd","--disk_radius",
                      type=float,default=100,
                      help="Protoplanetary disk radius, typically 100 AU")
  parser.add_argument("--adaptive_timestep",
                      action="store_true",
                      help="Use experimental adaptive timestep")
  parser.add_argument("-f","--filename",type=str,default="",help="Base name for files to SAVE, i.e. \"<filename>-yields.csv\", by default adopts the convention \"simulation-YY-MM-DD-HH-MM-SS\" based on sim start time")
  parser.add_argument("--no_massive_star_requirement",
                      action="store_true",
                      help="Do not require the formation of a massive star in the cluster (no re-rolls)")
  parser.add_argument("--star_min_mass",
                      type=float,default=0.01,
                      help="Minimum mass for stars in cluster, default uses full range of Maschberger distribution (0.01 msol)")
  parser.add_argument("--star_max_mass",
                      type=float,default=150.0,
                      help="Maximum mass for stars in cluster, default uses full range of Maschberger distribution (150 msol)")
  # Arguments governing AGB interlopers
  parser.add_argument("-i","--interloper",action="store_true",
                      help="Throw an interloping AGB star into the simulation")
  parser.add_argument("-mi","--interloper_mass",
                      type=float,default=3.0,
                      help="Mass of the interloping star, needs to be a valid mass")
  parser.add_argument("-rbi","--interloper_bubble_radius",
                      type=float,default=0.1,
                      help="Bubble size of interloping stars stellar wind, by default 0.1 pc")
  parser.add_argument("-ri","--interloper_radius",
                      type=float,default=None,
                      help="Interloper closest approach radius from barycenter in parsecs, assuming perfectly straight path, by default a random value between 0 pc and the star-forming region radius")
  parser.add_argument("-di","--interloper_distance",
                      type=float,default=None,
                      help="Interloper initial distance, by default this is 2*rc.")
  parser.add_argument("-vi","--interloper_velocity",
                      type=float,default=None,
                      help="Interloper initial velocity towards the cluster in km/s, again assumes perfectly straight apth to the closest approach in a straight line, by default a random value between 1 and 100km/s")
  parser.add_argument("-ti","--interloper_offset_time",
                      type=float,default=0.0,
                      help="Time until interloper enters AGB phase in Myr, by default this is zero, star enters AGB phase at the start of the simulation")
  parser.add_argument("-trji","--interloper_trajectory",
                      action="store_true",
                      help="Write AGB position to text file, agb_trajectory.dat")

  parser.add_argument("-t_f","--final_time",type=float,default=10.0,help="Final time to simulate to in Myr")
  parser.add_argument("-v","--verbose",action="store_true",help="Print additional statements")
  # Finish parser, and start running main function
  args = parser.parse_args()
  main(args)