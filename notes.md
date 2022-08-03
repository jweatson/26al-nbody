# Code Notes

## Implementation

### I/O

- Going to use argparse for inputs
- Input parameters:
  - Cluster mass
  - No. stars
- Outputs:
  - Plots in a separate folder
  - Updated simulation details 

## TODO
General progression of what needs to be done.
- [X] Get AMUSE working
  - Amuse working in a passable state on laptop, limitations on modules to to ARM processor
- [X] Test simple 3-body system to verify it works
  - In the end it was easier to make a King distribution
  - This needs to be looked at, potentially in 3D
  - Determiend how plotting functions work, it's a bit of a hack, it might make sense to use pure matplotlib rather than the extensions
  - Seems like the best way is to copy particle arrays after the fact
    - particles -> cluster.particles -> plot_particles
    - Something along those lines
- [X] Code stellar mass distribution
  - [X] Need to work out how mass distribution works in this case
    - Using Maschberger distribution, calculates distribution from stars and then feeds total mass into Plummer model, is this valid?
- [X] Code cluster distribution
  - [X] Use plummer model?
    - Stellar radius and number stars defined, mass is then calculated after mass distribution calculated
- [X] Model Evolution
  - [X] Regular steps
  - [X] Slow down simulation (100x slower?) when stars are close to massive star
    - [X] Get keys for each massive star
    - [X] Massive star check for proximity, 5x wind deposition radius
- [X] Star evolution
  - [X] Based on MESA model, interpolate between masses?
    - [X] Switched to SeBa model for consistency
  - [X] Mass loss rate only needs to be calculated during proximity event
    - [X] Doesn't need to be calculated, added long with continuous evolution and supernovae with SeBa
  - [X] Would be better to rip data from MESA using a separate script
    - Again, redundant, as using SeBa
- [X] Code wind mapping, if within a certain radii (say 1pc)
  - See Richards code for how it was done with supernovae
  - Lictenberg 2016 used a short range approximation, would be best considering wind density markedly lower
  - Skips fluid step
- [X] Disc condensation/decay, over a period of millions of years most discs would end up forming planets, and should be removed from the simulation
- [ ] Disc destruction, stripping of radius through too-close interaction
  - [ ] Not implemented, can probably do a too close for comfort supernova interaction
- [X] Wind mapping based on evolution on WR
  - [X] Use mesa/MIST model as lookup table/interpolation potentially
    - Good for mass loss and terminal velocity of winds
    - Assuming solar metallicity, rotating
  - [X] Use 26Al and 60Fe yields from Limongi & Chieffi
- [ ] Disk dynamics
- [X] Supernovae?
  - [X] Included

