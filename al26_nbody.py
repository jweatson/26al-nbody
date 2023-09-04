"""
26al-nbody.py
- This programme is meant to run as a 
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
# Plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns
# Math/Array libraries
import numpy as np
from numpy.random import uniform, exponential # Random distributions for IMF 
from math import exp, ceil, pi # Always useful
from scipy.interpolate import splev,splrep # Interpolation libraries for wind yield calculations
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

# GLOBAL VALUES
n_plot = 100 # Number of checkpoints to make
module_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
### MODELS
gravity_model = Hermite # Currently using the Hermite model as it runs fine single threaded on M1
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
    self.filename        = filename
    self.time            = []
    self.local_26al      = []
    self.global_26al     = []
    self.sum_local_26al  = []
    self.sum_global_26al = []
    self.local_60fe      = []
    self.global_60fe     = []
    self.sum_local_60fe  = []
    self.sum_global_60fe = []
    self.first_write     = True
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
    self.sum_local_26al.append(sum(list(cluster.mass_26al_local.value_in(msol))))
    self.sum_global_26al.append(sum(list(cluster.mass_26al_global.value_in(msol))))
    self.local_60fe.append(list(cluster.mass_60fe_local.value_in(msol)))
    self.global_60fe.append(list(cluster.mass_60fe_global.value_in(msol)))
    self.sum_local_60fe.append(sum(list(cluster.mass_60fe_local.value_in(msol))))
    self.sum_global_60fe.append(sum(list(cluster.mass_60fe_global.value_in(msol))))
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
      f.write("time,local_26al,global_26al,local_60fe,global_60fe\n")
  def write_to_csv(self):
    """
    Append to global yields file, this is done every step
    """
    with open("{}-cluster-yields.csv".format(self.filename),"a") as f:
      f.write("{:3e},{:3e},{:3e},{:3e},{:3e}\n".format(
        self.time[-1],
        self.sum_local_26al[-1],
        self.sum_global_26al[-1],
        self.sum_local_60fe[-1],
        self.sum_global_60fe[-1]
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

  if bar is None:
    print("! Saving checkpoint file \"{}\"...".format(state_filename),end=" ")
  # else:
  #   bar.set_description("Saving checkpoint #{}...".format(str(nfile).zfill(5)),refresh=True)

  t1 = time()
  # Pack all relevant structures into the state object
  state = State(cluster,converter,metadata) # Build state object
  # Initialise LZMA file and dump state into file
  with open(state_filename, "wb") as f:
    compressed_state = compress(pickle.dumps(state))
    f.write(compressed_state)
  # Done!
  t2 = time()

  if bar is None:
    print("Done! Took {:3f} seconds!".format(t2-t1))
  # else:
  #   bar.set_description("Saving checkpoint file \"{}\"... Done! Took {:3f} seconds!".format(state_filename,t2-t1),refresh=True)

  # Write yields object to disk
  if bar is None:
    print("! Saving yields file \"{}\"...".format(yields_filename),end=" ")
  # else:
  #   bar.set_description("Saving yields file \"{}\"...".format(yields_filename),refresh=True)
    
  t3 = time()
  yields.marinate(yields_filename)
  # Done!
  t4 = time()

  if bar is None:
    print("Done! Took {:3f} seconds!".format(t4-t3))
  else:
    bar.set_description("Saving checkpoint #{}... Done! Took {:3f} seconds!".format(str(nfile).zfill(5),t4-t1),refresh=True)
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
  mass_msol   = mass.value_in(msol)   # Convert star mass to float in solar mass
  masses_msol = masses.value_in(msol) # Convert masses to floats in solar masses
  yields_msol = yields.value_in(msol) # Convert yields to floats in solar masses

  # First check for minimum and maximum mass conditions for set
  if mass_msol < min(masses_msol) or mass_msol > max(masses_msol):
    slr_yield = 0.0 | msol
  else:
    spline    = splrep(masses_msol,yields_msol)       # Build scipy spline object
    slr_yield = float(splev(mass_msol,spline)) | msol # Fit spline
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
  # print(approx_lifespan.value_in(myr))
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
  print(module_directory)

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

def evolve_simulation(cluster,converter,gravity,stellar,yields,metadata,t_f,Rc,bar):
  """
  Evolve simulation with an adaptive timestep

  Timestep calculation:
  - Timestep is separate from N-body step, which is handled by the numerical solver
  - Timestep is adaptive, and evolves from two main conditions
    - Proximity and sudden intercept velocity with massive stars, timestep is determined by
      the determining the first low-mass star to collide with a massive star assuming it was
      travelling on a collision course at its current velocity, this is then divided by 10
      for good measure. This is a fairly fast calculation but can change timestep effectively
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
  # Generate a list of indices for high mass and low mass stars
  hm_id,lm_id = get_high_mass_star_indices(cluster)

  ### TIME-STEP ROUTINES
  # Get maximal timestep, only used in very sparse environments
  dt = t_f / n_plot
  dt_fine = dt / 100 # Fine timestep, for minimising
  # Calculate safe timestep, based on proximity of stars and their current velocities
  for i in hm_id:
    hm_star = gravity.particles[i]
    for j in lm_id:
      lm_star = gravity.particles[j]
      d = calcStarDistance(hm_star,lm_star)
      # If star were on a direct intercept course, calculate time to intercept 
      v_int = calc_star_vel(lm_star.vx,lm_star.vy,lm_star.vz)
      t_int = d / v_int
      # Reduce significantly to find timestep, used for variable step
      dt_p = 0.2 * t_int
      dt   = min(dt_p,dt)
      # Now compare compare to a much smaller timestep to prevent overrefinement and extremely small steps
      dt = max(dt,dt_fine)
  # From minimised dt, find new timestep
  t_new = t + dt

  # Check timestep is valid against finishing simulation or plotting
  finish = False
  # Check to see if simulation will finish
  if t_new > t_f:
    t_new = t_f
    dt = t_new - t
    finish = True

  ### N-BODY ROUTINES
  # Now, evolve the N-body simulation (most time consuming section of iteration)
  gravity.evolve_model(t_new)

  ### STELLAR EVOLUTION ROUTINES
  # First, evolve stars according to SEBA model
  stellar.evolve_model(t_new)
  ### STELLAR EVOLUTION - SUPERNOVA
  # Add kick velocity to stars gone supernova
  # This has to use array indexing because it requires participation from multiple datasets
  sn_id = []
  for i in hm_id:
    vx_k = stellar.particles[i].natal_kick_x
    if vx_k != 0.0 | kms:
      # Check to see if star has been kicked before, if it has, ignore it and move on
      if cluster[i].kicked == False:
        # Get other properties
        vy_k = stellar.particles[i].natal_kick_y
        vz_k = stellar.particles[i].natal_kick_z
        # Add value to velocities in N-body simulation
        gravity.particles[i].vx += vx_k
        gravity.particles[i].vy += vy_k
        gravity.particles[i].vz += vz_k
        # Ensure this is a one time thing by setting SN kick flag to false
        cluster[i].kicked = True
        # Add index to list of supernovae indices (for later)
        sn_id.append(i)

  ### DISK ROUTINES

  ## Calculate SLR deposition for each wind
  # There are probably more reasonable ways to do this, however there is a minimal speed difference and this is elaborated on for clarity
  for i in hm_id: # Iterate through high mass stars
    # Massive star parameters
    mdot = - stellar.particles[i].wind_mass_loss_rate
    for j in lm_id: # Iterate through low mass stars
      # Check to see if disk exists in low mass star system
      if t_new <= cluster[j].tau_disk:
        ### Parameters relevant to both models
        d = calcStarDistance(gravity.particles[i],gravity.particles[j])
        r_disk = cluster[j].r_disk
        # Low mass star velocities
        vx = gravity.particles[j].vx
        vy = gravity.particles[j].vy
        vz = gravity.particles[j].vz
        # Calculate the traversed distance of the low mass system, common between global and local
        disk_spd    = (vx**2 + vy**2 + vz**2)**0.5
        d_disk_trav = disk_spd * dt
        # Calculate cross section and wind absorption for global model
        eta_bub_global  = calc_eta_bubble_wind(r_disk,2*Rc,d_disk_trav)
        wind_abs_global = mdot * eta_bub_global * dt
        ## Calculate rates for each SLR for winds
        wind_26al_global = wind_abs_global * cluster[i].wind_ratio_26al
        wind_60fe_global = wind_abs_global * cluster[i].wind_ratio_60fe
        # Calculate new disk SLR mass for global model
        # For 26Al
        mass_26al_old_global = cluster[j].mass_26al_global
        mass_26al_new_global = mass_26al_old_global + wind_26al_global
        # For 60Fe
        mass_60fe_old_global = cluster[j].mass_60fe_global
        mass_60fe_new_global = mass_60fe_old_global + wind_60fe_global
        # For global model, rewrite values
        cluster[j].mass_26al_global = mass_26al_new_global
        cluster[j].mass_60fe_global = mass_60fe_new_global

        # Now calculate disk interaction for local model
        if d <= r_bub_local_wind:
          # Calculate cross section and wind absorption for local model
          eta_bub_local = calc_eta_bubble_wind(r_disk,r_bub_local_wind,d_disk_trav)
          wind_abs_local = mdot * eta_bub_local * dt
          # Calculate wind deposition rates
          wind_26al_local = wind_abs_local * cluster[i].wind_ratio_26al
          wind_60fe_local = wind_abs_local * cluster[i].wind_ratio_60fe
          # Calculate new disk SLR mass
          # For 26Al
          mass_26al_old_local = cluster[j].mass_26al_local
          mass_26al_new_local = mass_26al_old_local + wind_26al_local
          # For 60Fe
          mass_60fe_old_local = cluster[j].mass_60fe_local
          mass_60fe_new_local = mass_60fe_old_local + wind_60fe_local
          # Rewrite values for local model
          cluster[j].mass_26al_local = mass_26al_new_local
          cluster[j].mass_60fe_local = mass_60fe_new_local

  # If a supernova has occured, deposit al26 and fe60 from supernova
  # Supernova injection only increases mass and mix frac, does not change mix frac
  # While there is some redundant recalculation here, this is only called a handful of times per sim
  # and the overhead is fairly minimal

  for i in sn_id:
    al26_sn = cluster[i].al26_sn_yield
    fe60_sn = cluster[i].fe60_sn_yield
    for j in lm_id:
      d = calcStarDistance(gravity.particles[i],gravity.particles[j])
      if d < r_bub_local_sne:
        r_disc   = cluster[j].r_disc
        eta_disk = calc_eta_disk_sne(r_disc,d)
        al26_inj = al26_sn * eta_disk
        fe60_inj = fe60_sn * eta_disk
        # Calculate new values
        m_al26_old = cluster[j].mass_26al_sne
        m_fe60_old = cluster[j].mass_60fe_sne
        m_al26_new = m_al26_old + al26_inj
        m_fe60_new = m_fe60_old + fe60_inj
        # Write to cluster
        cluster[j].M_al26 = m_al26_new
        cluster[j].M_fe60 = m_fe60_new

  ### HOUSEKEEPING ROUTINES
  # Update progress bar using dt value
  # This is a fairly brute force way of doing it, but the progress bar has always been a bit
  # Inaccurate when it comes to indeterminate numbers of steps
  prog_old = bar.n
  prog_new = (t_new / t_f)
  prog_del = prog_new - prog_old
  bar.update(prog_del)
  # Update the conditions of metadata object
  metadata.update(t_new)
  # Update yields
  yields.update_state(gravity.model_time,cluster)
  # Save data to checkpoint
  filename = metadata.filename
  nfile = metadata.most_recent_checkpoint
  save_checkpoint(filename,nfile,cluster,converter,yields,metadata,bar=bar)

  # Finish simulation and return, if finish = False, then this will be run again inside a loop in
  # the main function

  return cluster,gravity,stellar,yields,metadata,finish,bar

def get_high_mass_star_indices(cluster):
  """
  Rather than creating subsets of data, which use a lot of memory and have synchronisation issues
  we instead determine which stars are massive or not, and returns the index, such that all 
  simulation objects can be addressed at once.
  While this is run at every step, it is a comparatively lightweight loop, and scales such that
  O(n)

  Inputs:
    - cluster: The cluster particule array, which has the high_mass flag for each star
  Outputs:
    - hm_id: an array of high mass stars indices
    - lm_id: an array of low mass star indices
  """
  hm_id = []
  lm_id = []
  for i,star in enumerate(cluster):
    if star.high_mass == True:
      hm_id.append(i)
    else:
      lm_id.append(i)
  return hm_id,lm_id

def initFile(filename,header):
  """
  Quick function to initialise file at start of simulation
  As data appended at the end to save on memory, this has to be run at the start of the
  """
  with open(filename,"w") as f:
    f.write(header)
  return

def massSubSet(dataset):
  """
  Get mass subsets for high mass and low mass stars
  This is sort of redundant, but it just looks a little cleaner on the evolve stars section
  There could be a way to determine whether a star is low or high mass
  """
  lm_stars = dataset.select(lambda m : m <  12.0 | msol, ["mass"])
  hm_stars = dataset.select(lambda m : m >= 12.0 | msol, ["mass"])
  return lm_stars, hm_stars

def Al26WindRatio(mass):
  """
  Calculate the ratio of Al26 in the wind of a massive star
  This is calculated by averaging the wind yield with the total mass loss of a star
  A 5th order polynomial fit is used to estimate the total wind yield, this is a more complex
  polynomial than the supernovae fits, however it fits the curve significantly better.
  Conversely, 5th order fits diverge too much for the supernovae fits at low masses.
  
  Function is based on data collected in:
    Limongi, M., & Chieffi, A. (2006).
    The Nucleosynthesis of 26Al and 60Fe in Solar Metallicity Stars Extending in Mass from 11
    to 120 M☉: The Hydrostatic and Explosive Contributions.
    The Astrophysical Journal, 647(1), 483. https://doi.org/10.1086/505164
  """

  # Hard coded the 5th order polynomial fit, was found to be the most accurate
  fit = np.asarray([+1.17105742e-08,
                    -4.00763571e-06,
                    +5.28909881e-04,
                    -3.42510723e-02,
                    +1.12603185e+00,
                    -1.97872519e+01])
  m_msol = mass.value_in(msol) 
  polyobj = np.poly1d(fit)
  # Find the total lifetime of the star, ahead of time
  evol = SeBa(number_of_workers=1)
  evol.particles.add_particle(Particle(mass=mass))
  # Evolve the star until it dies
  evol.evolve_model(1000. | myr)
  # Get the stars final mass
  final_mass = evol.particles[0].mass
  # Now, calculate the 
  m_loss_tot = (mass - final_mass).value_in(msol)
  wind_yield = 10**polyobj(m_msol)
  wind_ratio = wind_yield / m_loss_tot
  evol.stop()
  # Finish!
  return wind_ratio

def Al26SNYield(mass):
  """
  Calculate the Al26 supernova yield for a star of a specific mass
  A 3rd order polynomail fit is used to estimate the total SNe yield, this seems to provide a fairly
  accurate value for the supernova yield
  
  Function is based on data collected in:
    Limongi, M., & Chieffi, A. (2006).
    The Nucleosynthesis of 26Al and 60Fe in Solar Metallicity Stars Extending in Mass from 11
    to 120 M☉: The Hydrostatic and Explosive Contributions.
    The Astrophysical Journal, 647(1), 483. https://doi.org/10.1086/505164
  """

  # Hard coded third order polynomial, was found to be the most accurate
  fit = np.asarray([+1.83477307e-06,
                    -4.40040850e-04,
                    +3.81826726e-02,
                    -4.98382877e+00])
  m_msol   = mass.value_in(msol) 
  polyobj  = np.poly1d(fit)
  sn_yield = 10**polyobj(m_msol)
  return sn_yield | msol

def Fe60SNYield(mass):
  """
  Calculate the Fe60 supernova yield for a star of a specific mass
  A 3rd order polynomail fit is used to estimate the total SNe yield, this seems to provide a fairly
  accurate value for the supernova yield
  
  Function is based on data collected in:
    Limongi, M., & Chieffi, A. (2006).
    The Nucleosynthesis of 26Al and 60Fe in Solar Metallicity Stars Extending in Mass from 11
    to 120 M☉: The Hydrostatic and Explosive Contributions.
    The Astrophysical Journal, 647(1), 483. https://doi.org/10.1086/505164
  """

  # Hard coded third order polynomial, was found to be the most accurate
  fit = np.asarray([+6.55399558e-06,
                    -1.59217611e-03,
                    +1.25085986e-01,
                    -7.57683552e+00])
  m_msol   = mass.value_in(msol) 
  polyobj  = np.poly1d(fit)
  sn_yield = 10**polyobj(m_msol)
  return sn_yield | msol

def discLifeTime():
  """
  Calcualting a disk lifetime, we assume a mean disc lifetime of 5Myr, with an exponential 
  distribution to calculate the decay time.
  Calculating the decay time ahead of the simulation - effectively predetermining the fate of the
  disc - has a number of dire philosophical connotations, but is probably fine for an N-body
  simulation.

  This is based off of estimated lifetimes from:
    Richert, A. J. W., Getman, K. V., Feigelson, E. D., Kuhn, M. A., Broos, P. S., Povich, M. S.,
    Bate, M. R., & Garmire, G. P. (2018).
    Circumstellar disc lifetimes in numerous galactic young stellar clusters.
    Monthly Notices of the Royal Astronomical Society, 477, 5191–5206.
    https://doi.org/10.1093/mnras/sty949 
  """
  lam = 5.0 # Scale height, mean lifetime of 5Myr
  tau = exponential(lam) | myr
  return tau

def calc_v(vx,vy,vz):
  return ((vx*vx) + (vy*vy) + (vz*vz))**0.5

def calc_eta_bubble_wind(r_disk,r_bub,d_bub_trav):
  """
  Calculate the cross section of sweeped-up wind from a disk.
  As a disc traverses through a wind blown bubble, mass from the stellar wind is dredged up, thus
  deposition rate is a factor of the size of the bubble, the velocity between the stars and the
  size of the disc.
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

def calcStarDistance(star1,star2):
  """
  Calculates distance between star 1 and star 2, uses amuse values so unit is flexible.
  You should know this calculation.
  """

  x1,y1,z1 = star1.x,star1.y,star1.z
  x2,y2,z2 = star2.x,star2.y,star2.z
  d = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
  return d

def generateMasses(nstars):
  """
  Generate a series of masses for stars using the Maschberger IMF, ensures that at least 1 massive star is included
  """

  def maschberger(m,G_lower,G_upper):
    mu = 0.2 # Average star mass
    a  = 2.3 # Low mass component
    b  = 1.4 # High mass component
    A = (((1-a) * (1-b))/mu) * (1/(G_upper-G_lower))
    p = A * ((m / mu)**(-a)) * ((1 + (m/mu)**(1-a))**(-b))
    return p

  def maschberger_aux(m):
    mu = 0.2 # Average star mass
    a  = 2.3 # Low mass component
    b  = 1.4 # High mass component
    return (1 + ((m/mu)**(1-a)))**(1-b)

  m_lower  = 0.01
  m_upper  = 150
  G_lower  = maschberger_aux(m_lower)
  G_upper  = maschberger_aux(m_upper)
  pm_upper = maschberger(m_lower,G_lower,G_upper)
  pm_lower = maschberger(m_upper,G_lower,G_upper)
  masses   = []

  min_mass = 0.1 # Limit to red dwarfs
  max_mass = 50. # No supermassive stars 

  # Generate mass distribution 
  with tqdm(total=nstars) as pbar:
    while len(masses) < nstars:
      m  = uniform(min_mass,max_mass)
      pm = uniform(pm_lower,pm_upper)
      if pm < maschberger(m,G_lower,G_upper):
        masses.append(m)
        pbar.update(1)
  pbar.close()
  # Check to see if a high mass star is present
  if max(masses) < 13.0:
    print("No massive stars in cluster! Re-rolling!")
    masses = generateMasses(nstars)
  # Convert list into array
  masses = np.asarray(masses)
  return masses

def init_cluster(model, nstars, Rc, SLRs, nmass=3):
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

  print("Sampling masses...")
  # First, generate a series of masses
  masses = generateMasses(nstars)
  # Set basic simulation parameters
  Mcluster  = sum(masses) | msol
  print("Done! Generating cluster...")

  # Now create the cluster
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
    star.r_disk      = 100.0 | au # Stars have a common initial disk size
    star.tau_disk    = discLifeTime() # Calculate disk lifetime from random sample distribution
    # SLR properties
    ## Aluminium group
    star.mass_27al = 8.500e-6 * star.mass # Calculate stable Al mass from Chrondritic samples
    star.mass_26al_local  = 0.0 | kg
    star.mass_26al_global = 0.0 | kg
    star.mass_26al_sne    = 0.0 | kg 
    ## Iron group
    star.mass_56fe = 1.828e-4 * star.mass # Claculate stable Fe mass from Chrondritic samples
    star.mass_60fe_local  = 0.0 | kg
    star.mass_60fe_global = 0.0 | kg
    star.mass_60fe_sne    = 0.0 | kg 
    # Mass dependent based on bracketing (if a star is massive or not)
    if star.mass >= 13.0 | msol:
      star.high_mass = True
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
    elif star.mass < 13.0 | msol:
      star.high_mass = False
      star.disk_alive = True
      star.total_wind_loss = 0.0 | msol # Mass loss for low mass stellar winds can be neglected
    else:
      raise Exception("Somehow a star has a weird mass!")

  # Finish up and return cluster
  print("Cluster done!")
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
    cluster,converter = init_cluster("plummer",nstars,Rc,SLRs)
    used_checkpoint = False
  else:
    print("! Loading from file {}...".format(args.reload))
    # First, check
    if args.n_checkpoint == None:
      checkpoint_number = most_recent_checkpoint(args.reload)
    else:
      checkpoint_number = args.n_checkpoint
    cluster,converter,yields,metadata = load_checkpoint(args.reload,checkpoint_number)
    metadata.update(args.reload,increment_checkpoint = False) # Update the metadata but don't increment number
    used_checkpoint = True

  # Initialise N-Body simulation
  gravity = gravity_model(converter,number_of_workers=8)
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
  bar = tqdm(total = 1.0,desc="Simulation",position=0)
  # Begin simulation, open ended until finish is true
  finish = False
  while finish == False:
    cluster,gravity,stellar,yields,metadata,finish,bar = evolve_simulation(cluster,converter,gravity,stellar,yields,metadata,t_f,Rc,bar)
  print("!!! Finished !!!")
  gravity.stop()
  stellar.stop()
  return

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Calculate orbital trajectories and Al26 enrichment of a stellar cluster")
  parser.add_argument("-n", default=None, type=int, help="Number of stars in cluster")
  parser.add_argument("-rc", default=None, type=float, help="Cluster radius (pc)")
  parser.add_argument("-r","--reload", type=str, default="", help="Base name of files to RELOAD")
  parser.add_argument("-nc", "--n_checkpoint",type=int,default=None,help="Which checkpoint file to load, defaults to highest number")
  parser.add_argument("-m", "--model", type=str, default="plummer", help="Which model to use, defaults to Plummer sphere")
  parser.add_argument("-f","--filename",type=str,default="",help="Base name for files to SAVE, i.e. \"<filename>-yields.csv\", by default adopts the convention \"simulation-YY-MM-DD-HH-MM-SS\" based on sim start time")
  parser.add_argument("-t_f","--final_time",type=float,default=10.0,help="Final time to simulate to in Myr")
  # Finish parser, and start running main function
  args = parser.parse_args()
  main(args)