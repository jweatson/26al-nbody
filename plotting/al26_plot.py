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
from al26_nbody import State,Metadata,Yields,myr,pc,msol
import scipy
from amuse.units import units

def use_tex():
  plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern"
  })

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
  """
  yields = Yields("")
  yields.plate(filename)
  return yields

def check_interaction(xh,yh,zh,xl_arr,yl_arr,zl_arr,r):
  @nb.njit(parallel=True)
  def truth_table(xh,yh,zh,xl_arr,yl_arr,zl_arr,r):
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
  int_x = []
  int_y = []
  int_z = []
  truth = truth_table(xh,yh,zh,xl_arr,yl_arr,zl_arr,r)
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
    elif mass <= 3.0:
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
  ax.scatter(lm_x,lm_y,lm_z,marker="o",s=2.00,linewidth=0,alpha=0.50,color="tab:blue",label="$\\textrm{M}_\\star \leq 3 \\textrm{M}_\\odot$")

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
  ax.set_xlim(((-2*half_radius)+x_m,(2*half_radius)+x_m))
  ax.set_ylim(((-2*half_radius)+y_m,(2*half_radius)+y_m))
  ax.set_zlim(((-2*half_radius)+z_m,(2*half_radius)+z_m))
  ax.set_aspect("equal")
  ax.set_xlabel("X (pc)")
  ax.set_ylabel("Y (pc)")
  ax.set_zlabel("Z (pc)")
  leg = ax.legend(loc="upper left",markerscale=2)
  for line in leg.get_lines():
    line.set_linewidth(1.0)
  return ax




def calc_current_heating_rate(z_al,z_fe):
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