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

# 19th June 2023

- I'm back and working on this after working on the stuff from my previous paper.
- Moved entire build to Nemesis workstation, as it's now not being used to render planets

## Things that need doing
- [ ] Update deposition model to use Limongi + Cheffi 2018 rather than earlier paper
  - [ ] Class based system for stars, specifically using a dictionary for storing yields so it can be iterated
- [ ] Change file outputs
  - [ ] Using HDF5 files for data writing, updating tables every step

# Currrent work
- Right, painstaking file reading stuff done
- Need to integrate limongi calculation stuff properly
  - Just do it as-is, the processing for this is slow but its only for a handful of stars in the system, use if the mass-loss flag = 1
  - Calculate total mass loss first, then yields for each, and then ratios in the form of a class
  - Saves repeated calculation of mass loss through ageing, while also having all of the relevant statistics
- As for mismatching of isotopes and such, it might be a good idea to just print out the isotopes being managed

- I need to work on the slr depositing and emitting routines
- SLR routine should run on initialisation, be carried into the creation of the star class?
  - While this affects portability it is probably the best way to do it, as it does require it anyway, and prevents a tonne of code re-use
    - Best to do it that way then

- Ideally by the end of the day tomorrow (Wednesday) this should be finished, and then I can work what code needs editing work
  - Also need to test that writing these classes can fit into the particles system, because if it doesn't I am kind of fucked lmao
  - Cluster works but has to be iterated in a for loop, make sure that the thing is efficient then

- Need to confirm that this version is the most recent, edits were made on the laptop version of this to work with fractal clusters, but that seems to be the bulk of the changes
- Iirc the rest of it was in place and working, pending significant testing

# 18/7/23
- Code for SLR emission seems to work well now, need to work on:
  - Deposition
  - File saving
  - Decay calculations
  - Tying new class based system into code
  - Rewriting everything
- It's a shame things have been going rather slowly, but I think the tedious things might be out of the way

# 10/8/23
- Goddamn it, some issues cropped up
- I have to rip out the class system unfortunately, absolute pain
- However, the file saving issue is sovled, I can just use a 

## Tasks
- [X] Implement and verify that all of the properties work on initialisation
  - It should now be the case that adding additional elements should be fine, use 26al and 60fe for now?
  - There were a couple of sections that needed serious work anyway
- [ ] Get abundances for other elements (annoying, ask Richard, for a lead tomorrow)
- [ ] Check file saving
- [ ] Add reload from checkpoint system (bypass cluster system)
- [ ] Metadata dump
- [ ] Yield dump for each star, periodic
  - [ ] Ideally this is paged for each SLR, stored as a dictionary containing a dataset inside the class that becomes the pickle file
  - [ ] Slightly complex, real old lady swallowing a fly shit, but it should work and be easy to access
- [ ] Periodic checkpoints system
- [ ] Add supernovae yield calculator using more recent data

# 15/8/23
- Right, file saving works but is nowhere near fast or efficient enough
  - Best way of implementing:
    - Main thing is writing yield data, that should ideally be appended to disk
    - State of simulation should still be checkpointed through pickle
    - Individual star contributions could still be pickled at every checkpoint too, much less data dense
      - Stored in its own pickle for size
    - Store summation of all yields in text document

- It seems like the main thing is pickle dumping the data, it's not fast enough
- Using a combination of ubjson and pickle is best, pickle for complex datatypes, ubjson for yields data
  - This is extremely fast and effective
- [ ] Need to add a section to count up the number of checkpoints and store in the metadata file

# 18/8/23

Right, not much work done today, whoops.
Overall there isn't much to do on the lower levels now.
- [X] Write routine to find most recent file, or find it from arguments
- [X] Need to also recover base name to continue simulation, as well as most recent iteration
  - Kind of done
  - [X] Iterator based on most recent value (stored in metadata)
- [X] Rewrite arguments section to ignore mandatory inputs when loading from disk

Right, the general stuff is cleaned up, I'm happy with that, just using a series of conditional arguments in the arguments parser, seems liek the best way to do it 

Then its:

- [ ] Check and refactor execution loop
- [ ] Improve functions 
- [ ] Clear up old, unusued functions
- [ ] Clear up, make license and gitignore
  - [ ] Push contents
- [ ] Perform checks, devise some unit tests
- [ ] Need to show some graphical tests wrt constant injection etc. for the paper

16/09/23
- Right things work however I've determined that the disk routines are painfully slow, taking up about 98% of execution time after switching to faster N body codes
- They also scale very badly, as such, we gotta bust out jit

Jit can be used to make a very clunky, but fairly fast execution loop, however:
  - Certain sections of code have to be spun out as a series of numpy arrays, this should be comparatively fast, but would be a sticking point
  - High mass and low mass indices can be carried over, and used to determine which values get calculated, some slight memory overhead in terms of array sizes, and empty data, but that should work
  - Parallelise lm_id loop, have that be outer loop, its much faster that way
  - Afterwards, values can just be added back to the python array, I believe you can modify columns in amuse en-masse, but you have to test this first!
    - Create an array of ascending values, add to zeroed out column
  - So current execution loop:
    - Find lm_id and hm_id, already done
    - inputs:
      - hm_id
      - lm_id
      - vx,vy,vz
      - x,y,z
      - mdot
      - wind_ratio
      > **THIS NOW HAS TO BE DONE FOR EACH SLR**
      > **ALL OF THESE MUST BE DONE IN FLOATS**
      - (r disk is now a constant, don't adjust)