import pandas as pd
from numpy import polyfit,poly1d,log10,isfinite,linspace
import matplotlib.pyplot as plt
import sys
from scipy.interpolate import splrep,splev,CubicSpline,Akima1DInterpolator,pchip_interpolate,interp1d
import numpy as np

# Make sure everything is running in TeX mode
plt.rcParams.update({
  "text.usetex": True,
  "font.family": "serif",
  "font.serif": ["Computer Modern Roman"],
})



def fit_n_write(data,filename,masses,order=2):
  isos = ["Al26","Fe60"]
  with open(filename,"w") as file:
    # Write header
    file.write("vel,fe/h,isotope")
    for n in range(order+1):
      file.write(",a{}".format(n))
    file.write("\n")
    # Begin fitting
    for i,row in data.iterrows():
      vel = row.tolist()[0]
      feh = row.tolist()[1]
      iso = row.tolist()[2]
      losses = row.tolist()[3:]
      # print(losses)
      if iso in isos:
        if 0.0 not in losses:
          print(iso,losses)
          # fit = polyfit(masses,log10(losses),order)
          # fitt = poly1d(fit)

          # masses = [13,15,20,25,30,40,60,80,120]
          # tck = splrep(masses,losses)
          tck = Akima1DInterpolator(masses,log10(losses))


          print(masses)
          print(losses)
          for n,mass in enumerate(masses):
            print(mass,losses[n])
          x = linspace(masses[0],masses[-1],1000)
          y = 10**tck(x)

          label = "$^{"+iso[2:]+"}$"+iso[0:2]
          plt.semilogy(x,y,label=label)
          plt.scatter(masses,losses)
  return

def main():
  # Read in the main file
  tot_yield_data = pd.read_fwf("limongi-table-8.txt",
                               skiprows=22,
                               names=["vel","fe/h","isotope","13m","15m","20m","25m","30m","40m","60m","80m","120m"],
                               infer_nrows=1000)
  lm_w_yield_data = pd.read_fwf("limongi-table-9.txt",
                                skiprows=17,
                                names=["vel","fe/h","isotope","13m","15m","20m","25m","30m","40m","60m","80m","120m"],
                                infer_nrows=1000)
  # Create subsets
  # print(lm_w_yield_data)
  tot_yield_sub  = tot_yield_data[(tot_yield_data["vel"] == 300) & (tot_yield_data["fe/h"] == 0)] 
  lm_w_yield_sub = lm_w_yield_data[(lm_w_yield_data["vel"] == 300) & (lm_w_yield_data["fe/h"] ==0)]
  # print(tot_yield_data)
  # print(tot_yield_sub)
  # print(lm_w_yield_sub)
  sn_yield_sub = tot_yield_sub.iloc[:,3:] - lm_w_yield_sub.iloc[:,3:]
  # print(sn_yield_sub)
  sn_yield_sub = sn_yield_sub.fillna(0.)
  sn_yield_sub = pd.concat([lm_w_yield_sub.iloc[:,0:3],sn_yield_sub.iloc[:,0:4]],axis=1)
  # print(tot_yield_sub)
  # print(sn_yield_sub)

  wind_yield_sub = pd.concat([lm_w_yield_sub.iloc[:,0:7],tot_yield_sub.iloc[:,7:]],axis=1)
  # print(wind_yield_sub)

  # Plot data


  print(wind_yield_sub[wind_yield_sub.isotope == "Fe60"])

  # Initialise plotter
  plt.figure(figsize=(5,3))

  fit_n_write(wind_yield_sub,"wind-yield-fits.csv",masses=[13.,15.,20.,25.,30.,40.,60.,80.,120.])
  plt.grid(True, which="both", ls="dotted",alpha=0.3)
  plt.xlabel("Stellar mass, M$_\\star$ (M$_\\odot$)")
  plt.ylabel("Total wind SLR yield, M$_\\textrm{SLR}$ (M$_\\odot$)")
  plt.legend(loc="lower right")
  plt.savefig("wind-yields.pdf",bbox_inches="tight")

  # Initialise plotter
  plt.figure(figsize=(5,3))

  fit_n_write(sn_yield_sub,"sne-yield-fits.csv",masses=[13.,15.,20.,25.],order=2)
  plt.grid(True, which="both", ls="dotted",alpha=0.3)
  plt.xlabel("Stellar mass, M$_\\star$ (M$_\\odot$)")
  plt.ylabel("Total SNR SLR yield, M$_\\textrm{SLR}$ (M$_\\odot$)")
  plt.legend(loc="lower right")
  plt.savefig("sne-yields.pdf",bbox_inches="tight")
  plt.clf()

  # Initialise plotter
  plt.figure(figsize=(5,3))

  fit_n_write(tot_yield_sub,"tot-yield-fits.csv",masses=[13.,15.,20.,25.,30.,40.,60.,80.,120.],order=3)
  plt.grid(True, which="both", ls="dotted",alpha=0.3)
  plt.xlabel("Stellar mass, M$_\\star$ (M$_\\odot$)")
  plt.ylabel("Total SNR SLR yield, M$_\\textrm{SLR}$ (M$_\\odot$)")
  plt.legend(loc="lower right")
  plt.savefig("tot-yields.pdf",bbox_inches="tight")


  wind_yield_sub.to_csv("wind-yields.csv",index=False)
  sn_yield_sub.to_csv("sne-yields.csv",index=False)

  # Write resultant data to disk

  # print(sn_yield_sub)

  return 


main()