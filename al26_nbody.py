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
import argparse
from math import ceil, pi
import sys
# Amuse imports, complex library so only importing specific things
from amuse.units import units
from amuse.ic import plummer
from amuse.lab import nbody_system, Hermite, SeBa, Particle
# Plotting libraries
import matplotlib.pyplot as plt
from pandas import array
import seaborn as sns
# Math/Array libraries
import numpy as np
from numpy.random import uniform, exponential
# Others
from tqdm import tqdm # Good little progress bar

# GLOBAL VALUES
n_plot = 100 # Number of plots to make
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
r_bub = 0.1 | pc        # Wind blown bubble size

def evolveSimulation(cluster,gravity,stellar,t_f,bar):
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
  - Evolve discs
    - Check for wind/supernovae injection
    - Calculate mixing fractions, add mass
  - Write data
    - Write CSV files, 1 column for each star
      - Al26 Wind gain rate
      - Total Al26 mass
      - Total Fe60 mass
      - Al26/Al27 mixing ratio
      - Fe60/Fe56 mixing ratio
    - Plot stellar positions
      - High mass and low mass stars are divided into populations
      - 1000 plots per simulation, even timestepping

  Inputs:
  - cluster: contains parameters of each star, handles:
    - Disc parameters, radius, lifetime, whether they are extant
    - Storage of mass injection
    - Supernova kick confirmation
  - stellar: SeBa stellar evolution simulation, which handles:
    - Stellar evolution, of course!
    - Evolutionary phase (MS, PMS, SNRelic, etc.)
    - Mass loss rate
  - gravity: N-body solver, handles:
    - Kinematics
    - Also stores the current canonical time
  - t_f: Final time to evolve to (AMUSE units, Myr)
  - bar: TQDM progress bar
  Outputs:
  - All outputs and inputs are mirrored
  - Initially this was just going to recurse, but Python stack depth limitations are strict  
  """

  ### INITIALISATION
  # Get current time
  t = gravity.model_time
  # Generate a list of indices for high mass and low mass stars
  hm_id,lm_id = getHighMassStarsIndices(cluster)

  ### TIME-STEP ROUTINES
  # Get maximal timestep, only used in very sparse environments
  dt = t_f / n_plot
  # Calculate safe timestep, based on proximity of stars and their current velocities
  for i in hm_id:
    hm_star = gravity.particles[i]
    for j in lm_id:
      lm_star = gravity.particles[j]
      d = calcStarDistance(hm_star,lm_star)
      # If star is on a direct intercept course, calculate time to intercept 
      v_int = calc_star_vel(lm_star.vx,lm_star.vy,lm_star.vz)
      t_int = d / v_int
      # Reduce significantly to find timestep, used for variable step
      dt_p = 0.1 * t_int
      dt   = min(dt_p,dt)
  # From minimised dt, find new timestep
  t_new = t + dt

  # Check timestep is valid against finishing simulation or plotting
  plot   = False
  finish = False
  # Check to see if simulation will finish
  if t_new > t_f:
    t_new = t_f
    dt = t_new - t
    plot = True
    finish = True
  # else:
  #   # Check to see if plotting should occur
  #   t_fr = t / t_f
  #   t_np = (t_f / n_plot) * ceil(t_fr * n_plot)
  #   if t_new > t_np:
  #     t_new = t_np
  #     dt = t_new - t
  #     plot = True

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

  ### DISC ROUTINES
  for i in hm_id:
    al26_frac = cluster[i].al26_wind_frac
    mdot      = - stellar.particles[i].wind_mass_loss_rate
    vxh = gravity.particles[i].vx
    vyh = gravity.particles[i].vy
    vzh = gravity.particles[i].vz
    for j in lm_id:
      if t_new <= cluster[j].disc_lifetime:
        d = calcStarDistance(gravity.particles[i],gravity.particles[j])
        if d < r_bub:
          r_disc = cluster[j].r_disc
          # Calculate lm star velocity relative to hm star
          vxl = gravity.particles[j].vx
          vyl = gravity.particles[j].vy
          vzl = gravity.particles[j].vz
          v_star = calc_star_rel_vel(vxl,vyl,vzl,vxh,vyh,vzh)
          # Calculate dredge efficiency
          eta_bub = calc_eta_bubble(r_disc,r_bub,v_star,dt)        
          # Calculate injected mass onto disk
          dm_inj_dt = mdot * eta_bub * al26_frac
          # Determine new al26 mass, calculate new al26/al26 ratio
          m_27  = cluster[j].M_al27
          m_old = cluster[j].M_al26
          m_inj = dm_inj_dt * dt
          m_26  = m_old + m_inj
          mix   = m_26 / m_27
          # Write to cluster
          cluster[j].mdot_al26 = dm_inj_dt
          cluster[j].M_al26    = m_26
          cluster[j].mix_al    = mix
        else:
          # If star is not in perimeter, make sure that al26 mass gain rate is zero
          # Other parameters should take care of themselves
          cluster[j].mdot_al26 = 0.0 | msolyr
      else:
        # If disc is decayed, make sure that al26 mass gain rate is zero
        # Other parameters in cluster should take care of themselves
        cluster[j].mdot_al26 = 0.0 | msolyr

  # If a supernova has occured, deposit al26 and fe60 from supernova
  # Supernova injection only increases mass and mix frac, does not change mix frac
  # While there is some redundant recalculation here, this is only called a handful of times per sim
  # and the overhead is fairly minimal

  for i in sn_id:
    al26_sn = cluster[i].al26_sn_yield
    fe60_sn = cluster[i].fe60_sn_yield
    for j in lm_id:
      d = calcStarDistance(gravity.particles[i],gravity.particles[j])
      if d < 1.0 | pc:
        r_disc   = cluster[j].r_disc
        eta_disc = calcEtaDisc(r_disc,d)
        al26_inj = al26_sn * eta_disc
        fe60_inj = fe60_sn * eta_disc
        # Calculate new values
        m_al26_old = cluster[j].M_al26
        m_fe60_old = cluster[j].M_fe60
        m_al26_new = m_al26_old + al26_inj
        m_fe60_new = m_fe60_old + fe60_inj
        # Get stable isotope disc mass
        m_al27 = cluster[j].M_al27
        m_fe56 = cluster[j].M_fe56
        # Calculate mixing ratio
        mix_al = m_al26_new / m_al27
        mix_fe = m_fe60_new / m_fe56
        # Write to cluster
        cluster[j].M_al26 = m_al26_new
        cluster[j].M_fe60 = m_fe60_new
        cluster[j].mix_al = mix_al
        cluster[j].mix_fe = mix_fe

  ### PLOTTING ROUTINES
  # Plot simulation if needed
  # if plot:
  #   plotSimulationState(gravity,int(ceil(n_plot * t_fr)),t_new)
  
  ### DISK WRITE ROUTINES
  # Whilst it is fairly inefficient to open the file and write into it, it is less convoluted
  # than passing a bunch of files between iterations of this function
  writeLine("al26-wind-rate.csv",t_new,cluster.mdot_al26.value_in(msolyr))
  writeLine("al26-mass.csv",t_new,cluster.M_al26.value_in(msol))
  writeLine("fe60-mass.csv",t_new,cluster.M_fe60.value_in(msol))
  writeLine("al26-ratio.csv",t_new,cluster.mix_al)
  writeLine("fe60-ratio.csv",t_new,cluster.mix_fe)

  ### HOUSEKEEPING ROUTINES
  # Update progress bar using dt value
  # This is a fairly brute force way of doing it, but the progress bar has always been a bit
  # Inaccurate when it comes to indeterminate numbers of steps
  prog_old = bar.n
  prog_new = (t_new / t_f)
  prog_del = prog_new - prog_old
  bar.update(prog_del)
  # Finish simulation and return, if finish = False, then this will be run again inside a loop in
  # the main function
  return cluster,gravity,stellar,finish,bar

def writeLine(filename,t,data):
  """
  Quick function to append data to the end of a file
  """
  t_myr = t.value_in(myr)
  with open(filename,"a") as f:
    f.write("\n{:+.4E},".format(t_myr))
  with open(filename,"ab") as f:
    np.savetxt(f,data,fmt="%+.4E",delimiter=", ",newline=", ")
  return 

def getHighMassStarsIndices(cluster):
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
  evol = SeBa()
  evol.particles.add_particle(Particle(mass=mass))
  # Evolve the star until it dies
  evol.evolve_model(1000. | myr)
  # Get the stars final mass
  final_mass = evol.particles[0].mass
  # Now, calculate the 
  m_loss_tot = (mass - final_mass).value_in(msol)
  wind_yield = 10**polyobj(m_msol)
  wind_ratio = wind_yield / m_loss_tot
  print(m_loss_tot,wind_yield,wind_ratio)
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

def plotSimulationState(simulation,n,t):
  """
  Plots the particule positions in 3D space, nothing much more than that.
  """
  # Get the x y and z coordinates for each star, convert from AMUSE units to pc
  x=simulation.particles.x.value_in(pc) 
  y=simulation.particles.y.value_in(pc)
  z=simulation.particles.z.value_in(pc)
  # Configure the plot
  sns.set(style="darkgrid")
  fig = plt.figure()
  fig.suptitle("T = {:.2} Myr".format(t.value_in(myr)))
  # Plot
  ax = fig.add_subplot(111, projection = '3d')
  ax.scatter(x, y, z)
  ax.set_xlim([-3,3])
  ax.set_ylim([-3,3])
  ax.set_zlim([-3,3])
  # Save!
  plt.savefig("plots/"+str(n).zfill(5)+".png")
  plt.close()
  # Done!
  return

def calc_eta_bubble(r_disc,r_bub,v_star,dt):
  """
  Calculate the cross section of sweeped-up wind from a disc.
  As a disc traverses through a wind blown bubble, mass from the stellar wind is dredged up, thus
  deposition rate is a factor of the size of the bubble, the velocity between the stars and the
  size of the disc.
  This assumes that the mass loss rate of the wind blowing the bubble tapers off drastically
  at the edge of the bubble.
  The al26 yield can be calculated by multiplying the resultant value by the mass of al26 emitted
  by the star
  """
  # Calculate from the timestep the distance travelled through the bubble over the timestep
  d_trav = v_star * dt
  # Use this value to calculate the total wind efficiency
  eta_bub = 0.75 * (r_disc**2) * d_trav / (r_bub**3)
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

def calcEtaDisc(r,d):
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
  # Check to see if mass 
  if max(masses) < 12.0:
    print("No massive stars in cluster! Re-rolling!")
    masses = generateMasses(nstars)
  # Convert list into array
  masses = np.asarray(masses)
  return masses

def initCluster(model, nstars, Rc, nmass=3):
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
  cluster = plummer.new_plummer_model(nstars, converter)
  # Add generated masses to cluster
  cluster.mass = masses

  # Now finish adding parameters to the cluster
  # Calculate star quantities that are dependent on mass or randomly distributed
  for star in cluster:
    star.M_al27 = 8.500e-6 * star.mass # Calc stable Al mass from Chrondritic samples
    star.M_fe56 = 1.828e-4 * star.mass # Calc stable Fe mass from Chrondritic samples
    # Calculate disc lifetime from random sampling of exponential distribution
    star.disc_lifetime = discLifeTime()

  # Determine if stars are high or low mass at ZAMS
  for star in cluster:
    if star.mass >= 12.0 | msol:
      star.high_mass      = True  # Star is high mass
      star.disc_alive     = False # Star has a disc
      star.al26_wind_frac = Al26WindRatio(star.mass) # Mass fraction of wind in star (MAl26/Mwind)
      star.al26_sn_yield  = Al26SNYield(star.mass)   # Al26 yield from supernova (Msol)
      star.fe60_sn_yield  = Fe60SNYield(star.mass)   # Fe60 yield from supernova (Msol)
    else:
      star.hi_mass        = False # Star is NOT high mass
      star.disc_alive     = True  # Star does not have a disc
      star.al26_wind_frac = 0.0   # Being a low mass star, wind does not produce Al26
      star.al26_sn_yield  = 0.0 | msol # Similarly, star does not undergo supernova
      star.fe60_sn_yield  = 0.0 | msol # Similarly, star does not undergo supernova

  # Initialise other properties, consistent across all stars
  cluster.radius    = 0.   | pc     # Stars themselves are dimensionless
  cluster.r_disc    = 100. | au     # Star wth a disk size of 100 AU
  cluster.mdot_al26 = 0.0  | msolyr # Current rate of Al26 injection through winds only
  cluster.mix_al    = 0.0           # Al26/Al27 mass ratio 
  cluster.mix_fe    = 0.0           # Fe60/Fe56 mass ratio 
  cluster.M_al26    = 0.0  | kg     # Total mass of Al26 adsorbed
  cluster.M_fe60    = 0.0  | kg     # Total mass of Fe60 adsorbed
  cluster.kicked    = False         # Supernova kick flag

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
  nstars = int(args.nstars) # Number of stars
  Rc     = args.Rc | pc  # Plummer sphere half-mass-radius
  t_f    = 10.     | myr # Final simulation time in Myr
  # Initialise star cluster and star masses
  cluster,converter = initCluster("plummer",nstars,Rc)
  # Initialise N-Body simulation
  gravity = gravity_model(converter)
  gravity.particles.add_particles(cluster)
  # Initialise stellar evolution simulation
  stellar = stellar_model()
  stellar.particles.add_particles(cluster)
  # Initialise progress bar
  bar = tqdm(total = 1.0)
  # Initialise write files
  initFile("al26-wind-rate.csv","AL26_WIND_RATE")
  initFile("al26-mass.csv","AL26_MASS")
  initFile("fe60-mass.csv","FE60_MASS")
  initFile("al26-ratio.csv","AL26_RATIO")
  initFile("fe60-ratio.csv","FE60_RATIO")
  # Begin simulation, open ended until finish is true

  print(stellar.particles.wind_mass_loss_rate)

  finish = False
  while finish == False:
    cluster,gravity,stellar,finish,bar = evolveSimulation(cluster,gravity,stellar,t_f,bar)
  print("!!! Finished !!!")
  return

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Calculate orbital trajectories and Al26 enrichment of a stellar cluster")
  parser.add_argument("nstars", type=float, help="Number of stars in cluster")
  parser.add_argument("Rc", type=float, help="Cluster radius (pc)")
  parser.add_argument("-m", "--model", type=str, default="plummer", help="Which model to use, defaults to Plummer sphere")
  args = parser.parse_args()
  main(args)