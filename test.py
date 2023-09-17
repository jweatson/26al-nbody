import al26_nbody
from amuse.units import units
import sys


# beepo = al26_nbody.read_SLRs("slr-abundances.csv")

cluster,converter  = al26_nbody.initCluster("plummer",100,1 | units.parsec)

# cluster.star_properties  = al26_nbody.Star(cluster.mass)
print(cluster.mass[0])

module_directory = sys.path[0]
SLRs = al26_nbody.read_SLRs(module_directory+"/slr-abundances.csv")


for star in cluster:
    star.properties = al26_nbody.Star(star.mass,SLRs)

# for star in cluster:
    # print(star.properties.SL)

# al26_nbody.calc_lifetime(cluster)


