import sys
sys.path.append("../")
from al26_nbody import discLifeTime
from amuse.lab import units
from numpy import empty
import matplotlib.pyplot as plt

arr = empty(100000)
for i in range(len(arr)):
  arr[i] = discLifeTime().value_in(units.Myr)

plt.hist(arr,bins=100)
plt.yscale("log")
plt.savefig("disclifetime.pdf")