import sys
sys.path.append("../")
from al26_nbody import initCluster
from amuse.lab import units
import matplotlib.pyplot as plt

clusters = []

cluster_sizes = [0.3,1.0,3.0]
cluster_types = ["fractal"]

fig = plt.figure(figsize=(10,10),dpi=300)
nrows = len(cluster_types)
ncols = len(cluster_sizes)

n = 1
for cluster_type in cluster_types:
  for cluster_size in cluster_sizes:
    print(cluster_type)
    cluster,converter = initCluster(cluster_type,1000, cluster_size | units.pc)
    x = cluster.x.value_in(units.pc)
    y = cluster.z.value_in(units.pc)
    c = cluster.mass.value_in(units.MSun)

    ax = fig.add_subplot(nrows,ncols,n)
    ax.set_xlabel("X (pc)")
    ax.set_ylabel("Y (pc)")
    ax.set_title("{} @ {:.1f} pc".format(cluster_type,cluster_size))
    ax.set(aspect=1)
    ax.scatter(x,y,c=c)

    n += 1

# fig.colorbar()
plt.savefig("clusters.png")