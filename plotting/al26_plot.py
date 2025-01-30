"""
Plotting library for al26 nbody paper
"""

import pickle
import ubjson
import zstandard as zstd
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import numba as nb
import sys,os
plotting_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
al26_nbody_dir = plotting_dir+"/../"
from al26_nbody import State,Metadata,Yields,myr,pc,msol,get_high_mass_star_indices
import scipy
from amuse.units import units
from numba import njit,prange
import pandas as pd


def use_tex(use_mnras=False):
  plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern"
  })

  if use_mnras == True:
    import matplotlib.font_manager
    SMALL_SIZE = 9
    MEDIUM_SIZE = 9
    BIGGER_SIZE = 9
    matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
    plt.rc('axes',titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes',labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick',labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick',labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend',fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure',titlesize=BIGGER_SIZE)

def read_state(filename):
  """
  A subset of the file loading code in load_checkpoint, loads a `state` file, first decompressing serialised data into memory (through the pickle library), and then converting it back into an object of the `State` class
  """
  c = zstd.ZstdDecompressor()
  with open(filename,"rb") as f:
    compressed = f.read()
    decompressed = c.decompress(compressed)
    state = pickle.loads(decompressed)
  return state

def read_yields(filename):
  """
  A little wrapper for the Yields class, since Yields has its own file importing code (Yields.plate), we'll just use that
  This also converts everything into numpy arrays, because its faster to do this now
  """
  yields = Yields("")
  yields.plate(filename)
  # Convert lists in yields into numpy arrays
  for attr, value in yields.__dict__.items():
    if type(yields.__dict__[attr]) == list:
      yields.__dict__[attr] = np.asarray(yields.__dict__[attr])

  return yields

def read_interloper_trajectory(filename):
  colnames=["sim_time","agb_time","x","y","z","bary_dist"]
  int_traj = pd.read_csv(filename,names=colnames,header=None)
  return int_traj

@nb.njit(parallel=True)
def check_interaction_truth_table(xh,yh,zh,xl_arr,yl_arr,zl_arr,r):
  nstars = len(xl_arr)
  truth  = np.zeros(nstars)
  for i in nb.prange(nstars):
    xl = xl_arr[i]
    yl = yl_arr[i]
    zl = zl_arr[i]
    rr = ((xl-xh)**2 + (yl-yh)**2 + (zl-zh)**2)**0.5
    if rr < r:
      truth[i] = 1
  return truth

def check_interaction(xh,yh,zh,xl_arr,yl_arr,zl_arr,r):
  int_x = []
  int_y = []
  int_z = []
  truth = check_interaction_truth_table(xh,yh,zh,xl_arr,yl_arr,zl_arr,r)
  for i,n in enumerate(truth):
    if n == 1:
      int_x.append(xl_arr[i])
      int_y.append(yl_arr[i])
      int_z.append(zl_arr[i])
      # Occlude (remove) matching values in array
      # del xl_arr[i]
      # del yl_arr[i]
      # del zl_arr[i]
  return int_x,int_y,int_z,xl_arr,yl_arr,zl_arr
    
def sphere_wireframe(x,y,z,r):
  """
  Draw sphere of radius r at x,y,z coordinates
  """
  # draw sphere
  u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:20j]
  xx = r*(np.cos(u)*np.sin(v))
  yy = r*(np.sin(u)*np.sin(v))
  zz = r*(np.cos(v))
  return xx+x,yy+y,zz+z

def plot_positions(particles,metadata,interaction_radius=0.1,ax=None):
  """
  Plot positions and interactions between high and low mass stars
  """
  x = particles.x.value_in(pc)
  y = particles.y.value_in(pc)
  z = particles.z.value_in(pc)
  masses = particles.mass.value_in(msol)
  disks = particles.disk_alive
  t = metadata.time.value_in(myr)
  half_radius = metadata.args.rc
  # Find average position of stars, used to move axes slowly over time
  x_m = np.mean(x)
  y_m = np.mean(y)
  z_m = np.mean(z)
  # Create lists to bin stars
  lm_x,lm_y,lm_z = [],[],[]
  im_x,im_y,im_z = [],[],[]
  hm_x,hm_y,hm_z = [],[],[]
  # Bin stars, this could probably be done using pandas, but this is fast enough (<1 second for 10,000 stars)
  for i,mass in tqdm(enumerate(masses)):
    if mass >= 13.0:
      hm_x.append(x[i])
      hm_y.append(y[i])
      hm_z.append(z[i])
    elif mass <= 13.0:
      if disks[i] == True:
        lm_x.append(x[i])
        lm_y.append(y[i])
        lm_z.append(z[i])
      else:
        im_x.append(x[i])
        im_y.append(y[i])
        im_z.append(z[i])

  # Now these are all done, plot the figure!
  # First check to see if a subplot has been used as an argument
  if ax is None:
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(projection="3d")

  ax.scatter(hm_x,hm_y,hm_z,marker="D",s=4.00,linewidth=0,alpha=1.00,color="tab:orange",label="$\\textrm{M}_\\star \geq 13 \\textrm{M}_\\odot$")
  ax.scatter(im_x,im_y,im_z,marker="o",s=2.00,linewidth=0,alpha=0.50,color="tab:blue",label="$\\textrm{M}_\\star \leq 3 \\textrm{M}_\\odot$")
  ax.scatter(lm_x,lm_y,lm_z,marker="o",s=2.00,linewidth=0,alpha=1.00,color="red",label="Disk")

  # Draw spheres and intersections
  interactions = 0
  for i in range(len(hm_x)):
    xx,yy,zz = sphere_wireframe(hm_x[i],hm_y[i],hm_z[i],interaction_radius)
    if i == 0: label = "SOI"
    else: label = ""
    ax.plot_wireframe(xx,yy,zz,color="r",linewidth=0.1,linestyle="dotted",label=label)
    int_x,int_y,int_z,lm_x,lm_y,lm_z = check_interaction(hm_x[i],hm_y[i],hm_z[i],lm_x,lm_y,lm_z,interaction_radius)
    interactions += len(int_x)
    ax.scatter(int_x,int_y,int_z,marker="o",s=2.00,linewidth=0,alpha=1.00,color="tab:blue")
    for j in range(len(int_x)):
      if i == 0 and j == 0: label = "Interaction"
      else: label = ""
      ax.plot([hm_x[i],int_x[j]],[hm_y[i],int_y[j]],[hm_z[i],int_z[j]],color="r",linewidth=0.2,label=label)
    print("Found {} interactions for star {}".format(len(int_x),i))

  # ax.scatter(im_x,im_y,marker="o",s=0.50,alpha=0.20,label="Intermediate Mass")

  ax.set_title("t = {:.2f} Myr, {} interacting stars".format(t,interactions))
  ax.set_xlim(((-10*half_radius)+x_m,(10*half_radius)+x_m))
  ax.set_ylim(((-10*half_radius)+y_m,(10*half_radius)+y_m))
  ax.set_zlim(((-10*half_radius)+z_m,(10*half_radius)+z_m))
  ax.set_aspect("equal")
  ax.set_xlabel("X (pc)")
  ax.set_ylabel("Y (pc)")
  ax.set_zlabel("Z (pc)")
  leg = ax.legend(loc="upper left",markerscale=2)
  for line in leg.get_lines():
    line.set_linewidth(1.0)
  return ax

def calc_current_heating_rate(z_al,z_fe):
  """
  This has not been finished yet!
  """
  H_al26 = 0.3551
  H_fe60 = 0.0396
  f_al = 8500e-6
  f_fe = 1828e-4
  Q_al = z_al * f_al * H_al26
  Q_fe = z_fe * f_fe * H_fe60
  Q_T = Q_al + Q_fe
  return Q_T

def calc_cdf(data,ax=None):
  x = np.sort(data)
  y = 1. * np.arange(len(data)) / (len(data) - 1)
  return x,y
  
def get_digit_from_filename(filename,length=5):
  digits = str(''.join(filter(str.isdigit, filename)))
  digit = digits[-length:]
  return digit

def calc_disk_final_enrichment(yields_data,lifetimes):
  """
  General post-processing function to find the enrichment at the final

  This function processes all yield columns, and appends additional columns suffixed with "_final"

  As this works by checking yield snapshots, an interpolation step is performed for each star in order to determine a more accurate value
  """

  from scipy.interpolate import Akima1DInterpolator
  nstars = len(lifetimes)
  t = yields_data.time
  # Best way of doing this is unfortunately messing around with attributes and keys
  isos   = ["26al","60fe"]
  models = ["global","local","sne"]
  # Iterate through models and isotopes
  for iso in isos:
    for model in models:
      key = model + "_" + iso
      y  = getattr(yields_data,key)
      fy = []
      # Iterate through stars in list, find lifespan, interpolate yield for particular case
      for i in range(nstars):
        interp = Akima1DInterpolator(t,y[:,i])
        tau = float(lifetimes[i])
        yy = float(interp(tau))
        # Special condition for disks with lifetime greater than simulation
        if np.isnan(yy) or tau >= t[-1]:
          yy = y[-1,i]   
        fy.append(yy)
      # Copy list back to object
      setattr(yields_data,key+"_final",fy)
  return yields_data

def calc_sn_times(initial_cluster,return_keys = False):
  """
  Calculate supernova time and mass, used for a plotting script, subsets the cluster so is faster but difficult to index with
  """
  from amuse.lab import SeBa
  # Make subset with massive stars  
  high_mass_cluster = initial_cluster[initial_cluster.mass >= (13.0 | msol)]
  # Fire up stellar evolution code
  stellar = SeBa()
  stellar.particles.add_particle(high_mass_cluster)
  # Enable runtime stop based on supernovae flag
  detect_supernova = stellar.stopping_conditions.supernova_detection
  detect_supernova.enable()

  # Get sorted list of supernovae masses and keys, in inverse mass order
  sn_times,sn_masses,sn_keys = [],[],[]
  sn_mass_sort_i = np.flip(np.argsort(high_mass_cluster.mass.value_in(msol)))
  for i in sn_mass_sort_i:
    sn_masses.append(high_mass_cluster.mass[i].value_in(msol))
    sn_keys.append(high_mass_cluster.key[i])

  # Begin evolving model to find supernovae lifetimes
  t = 0.0|myr
  while t <= 99.0 | myr:
    stellar.evolve_model(100.0 | myr)
    t = stellar.model_time
    sn_times.append(t.value_in(myr))

  del(sn_times[-1])
  stellar.stop()

  # Slight bodge here to keep compatibility with some old code
  if return_keys:
    return sn_times,sn_masses,sn_keys
  else:
    return sn_times,sn_masses

def calc_etot(state):
  """
  Fairly slow method, since these values are derived from Bhtree, which has to be initialised
  """
  from amuse.community.bhtree import Bhtree
  gravity = Bhtree(state.converter)
  gravity.particles.add_particles(state.cluster)
  eki = gravity.kinetic_energy
  epi = gravity.potential_energy
  eety = eki + epi
  gravity.stop()
  # eeky = state.cluster.kinetic_energy()
  # eepy = state.cluster.potential_energy()
  # eety = eeky + eepy
  return eety

def calc_dE(eti,et):
  dE = (eti-et)/et
  return dE

def calc_local_densities_nonumba(cluster):
  ftp    = 4.18879020479 # Four-thirds pi
  nstars = len(cluster)  # Number of stars in the cluster
  local_densities = np.zeros(nstars) # Local densities array
  # Calculate a matrix of distances squared, using amuse cluster class function
  d2 = cluster.distances_squared(cluster) 
  for i in range(nstars):
    # Slice array
    d2_slice  = d2[i,:]
    # Find indices to sort array
    d2_sort_i = np.argsort(d2_slice)
    # Get indices of 10 nearest stars
    d2_sort_i_nearest = d2_sort_i[1:11]
    # Calculate sphere mass, feed indices directly into array then sum, fastest way
    mass = cluster.mass[d2_sort_i_nearest].sum()
    # Calculate radius cubed directly, again, slightly faster
    r3 = np.power(d2_slice[d2_sort_i[-1]],1.5)
    # Calculate density, using precalculated four thirds pi value, convert to a float instead of a unit
    rho = (ftp * mass / r3).value_in(msol/pc**3)
    # Add to list
    local_densities[i] = rho
  return local_densities

@njit(parallel=True)
def local_densities_numba(x,y,z,masses):
  # Constants
  ftp = 4.18879020479 # Four-thirds pi, it's used a lot
  nstars = len(x)     # Number of stars in cluster
  # Variables
  local_rhos = np.zeros(nstars)  # Local densities
  d  = np.zeros((nstars,nstars)) # Distance matrix, 
  # Calculate distance matrix, faster to do it here than use Amuse function
  for i in prange(nstars):
    xi,yi,zi = x[i],y[i],z[i]
    for j in range(nstars):
      xj,yj,zj = x[j],y[j],z[j]
      d2 = (xi-xj)**2 + (yi-yj)**2 + (zi-zj)**2
      d[i,j] = np.sqrt(d2)
  # Now sit on each star and calculate local density
  for i in prange(nstars):
    # Slice array
    d_star = d[i]
    # Find indices for density calculation
    d_sort_idx = np.argsort(d_star)
    idx_nr = d_sort_idx[1:11] # 10 nearest stars indices
    idx_10 = idx_nr[-1]       # 10th nearest stars index (important difference)
    d_10   = d_star[idx_10]   # 10th nearest star distance (used for volume)
    # Calculate mass of 10 nearest stars, which reside in local density sphere
    mass = 0.0
    for j in idx_nr:
      mass += masses[j]
    # Calculate volume of local density sphere
    vol = ftp * d_10 * d_10 * d_10
    # Calculate local density
    local_rho = mass / vol
    # Add to local rhos array
    local_rhos[i] = local_rho
  # Calculate the local density and return
  return local_rhos

def calc_local_densities(cluster):
  """
  Calculate the local densities for all stars in a fractal cluster
  """
  # Split arrays before entering the JIT zone
  x      = cluster.x.value_in(units.pc)
  y      = cluster.y.value_in(units.pc)
  z      = cluster.z.value_in(units.pc)
  masses = cluster.mass.value_in(units.MSun)
  local_rhos = local_densities_numba(x,y,z,masses)
  return local_rhos

@njit()
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

@nb.njit(parallel=True)
def calc_wind_abs(lm_id_arr,hm_id_arr,
                  x_arr,y_arr,z_arr,
                  vx_arr,vy_arr,vz_arr,
                  mdot_arr,
                  wind_ratio_arr,
                  rdisk_arr,
                  distance_limit,
                  bubble_radius,
                  dt):
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

def calc_global_model_yield(cluster,stellar,time,dt,radius_method="halfmass"):
  """
  Calculate the global model yield rates for the current state of the simulation
  Whilst this is less accurate due to the lower temporal resolution than calculating it
  while the simulation is running, it should be good enough for most things
  """  
  # cluster = state.cluster
  # time    = state.metadata.time
  dt_mks  = dt.value_in(units.s)
  # Build stellar evolution model and evolve to snapshot time for mass loss rates
  stellar.evolve_model(time)
  # Calculate the half mass radius of the simulation
  if radius_method == "halfmass":
    half_mass_radius = calc_cluster_half_mass(cluster) | pc
    cluster_radius = (2.0 * half_mass_radius).value_in(units.km)
  elif radius_method == "virial":
    cluster_radius = cluster.virial_radius().value_in(units.km)
    # print((cluster_radius|units.km).value_in(pc))
  else:
    print("INVALID RADIUS METHOD")
    sys.exit()
  # Get indices of high and low mass stars
  hm_id,lm_id = get_high_mass_star_indices(cluster)
  # Split arrays such that they can be read into a NUMBA compliant function (pandas splitting is efficient, accelerated)
  x_arr     = cluster.x.value_in(units.km)
  y_arr     = cluster.y.value_in(units.km)
  z_arr     = cluster.z.value_in(units.km)
  vx_arr    = cluster.vx.value_in(units.km/units.s)
  vy_arr    = cluster.vy.value_in(units.km/units.s)
  vz_arr    = cluster.vz.value_in(units.km/units.s)
  mdot_arr  = - stellar.particles.wind_mass_loss_rate.value_in(units.kg/units.s)
  rdisk_arr = cluster.r_disk.value_in(units.km)
  wind_ratio_26al_arr = cluster.wind_ratio_26al
  wind_ratio_60fe_arr = cluster.wind_ratio_60fe

  if len(hm_id) == 0 or len(lm_id) == 0:
    wind_abs_global_26al = np.zeros(len(cluster))
    wind_abs_global_60fe = np.zeros(len(cluster))
  else:
    wind_abs_global_26al = calc_wind_abs(lm_id,hm_id,
                                        x_arr,y_arr,z_arr,
                                        vx_arr,vy_arr,vz_arr,
                                        mdot_arr,wind_ratio_26al_arr,rdisk_arr,
                                        distance_limit=0.0,
                                        bubble_radius=cluster_radius,
                                        dt = dt_mks) | units.kg
    wind_abs_global_60fe = calc_wind_abs(lm_id,hm_id,
                                        x_arr,y_arr,z_arr,
                                        vx_arr,vy_arr,vz_arr,
                                        mdot_arr,wind_ratio_60fe_arr,rdisk_arr,
                                        distance_limit = 0.0,
                                        bubble_radius=cluster_radius,
                                        dt = dt_mks) | units.kg
    wind_abs_global_26al = wind_abs_global_26al.value_in(msol)
    wind_abs_global_60fe = wind_abs_global_60fe.value_in(msol)
  # print(time,wind_abs_global_26al)
  return wind_abs_global_26al, wind_abs_global_60fe, stellar

def calc_star_distance(star1,star2):
  """
  Calculates distance between star 1 and star 2, uses amuse values so unit is flexible.
  You should know this calculation.
  """
  x1,y1,z1 = star1.x,star1.y,star1.z
  x2,y2,z2 = star2.x,star2.y,star2.z
  d = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
  return d

def calc_eta_disk_sne(r,d):
  # Constants
  cos60    = 0.5 # Hard-coding cos(60)
  eta_cond = 0.5 # Condensation efficiency
  eta_inj  = 0.7 # Injection efficiency
  # Now calculate eta_geom
  eta_geom = (cos60 * r**2) / (4 * d**2)
  # Calculate the whole thing
  eta_total = eta_cond * eta_inj * eta_geom
  return eta_total
  


