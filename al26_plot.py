"""
Plotting library for al26 nbody paper
"""

import pickle
import ubjson
import zstandard as zstd
from al26_nbody import State,Metadata,Yields,myr,pc

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

def plot_positions(particles):
  x = particles.x.value_in(pc)
  y = particles.y.value_in(pc)
  z = particles.z.value_in(pc)
  return x
  

state = read_state("test-1/test-state-00000.pkl.zst")
yields = read_yields("test-1/test-yields.ubj.zst")

import numpy as np

print(sum(plot_positions(state.cluster)))

