import sys
import os
script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append(script_dir)
sys.path.append(script_dir+"/../")

from al26_plot import get_digit_from_filename,read_state,plot_positions
from al26_nbody import State,Metadata
import matplotlib.pyplot as plt

n = sys.argv[1]
digit = get_digit_from_filename(n)
print(n)
print(digit)

print("Test",n,os.path.exists(n))
print(State)
state = read_state(n)

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern"
})

plot_positions(state.cluster,state.metadata)
filename = "pos-{}.png".format(digit)

plt.savefig("pos-{}.png".format(digit),dpi=600)
print("Finished {}".format(filename))