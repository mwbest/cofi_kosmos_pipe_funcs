"""
Functions that work to identify spectral features, and fit them for
wavelength calibration. Only meant for an specific use in our pipeline (Do not use them).
"""

import ipywidgets as widgets
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline, interp1d
from specutils import Spectrum1D
from specutils.manipulation import FluxConservingResampler, gaussian_smooth
from specutils.utils.wcs_utils import air_to_vac as a2v
import os
import pandas as pd
import glob
import astropy.units as u
from astropy.table import Table


__all__ = ['aprox_points', 'wave_cal']


def air_to_vac(spec):
    """
    Simple wrapper for the `air_to_vac` calculation within `specutils.utils.wcs_utils`

    Parameters
    ----------
    spec : Spectrum1D object

    Returns
    -------
    Spectrum1D object with spectral_axis converted from air to vaccum units

    """
    new_wave = a2v(spec.wavelength)
    outspec = Spectrum1D(spectral_axis=new_wave,
                         flux=spec.flux,
                         uncertainty=spec.uncertainty
                         )
    return outspec


def aprox_points(neon_spectra,neon_xpt, npixels=10):  
    npixels = 10    #This is the area around your guess that the new approximation will search for a peak within
    axis = neon_spectra.spectral_axis
    ne_spectrum = neon_spectra.flux
    
    #In order for the indexing to work in the next step, we need to make a list of the nearest integer to our guessed xpts
    integer_xpts = []
    for x in neon_xpt:
      integer_xpts.append(int(x-x%1))
    
    improved_xpts = [np.average(axis[g-npixels:g+npixels],weights=ne_spectrum[g-npixels:g+npixels] - np.median(ne_spectrum))
                             for g in integer_xpts]
    
    #print improved and original xvals to compare:
    print(improved_xpts)
    print(neon_xpt)
    
    #Checking how well it lines up
    plt.figure(figsize = (10, 4))
    plt.plot(neon_spectra.spectral_axis, neon_spectra.flux)
    for val in improved_xpts:
      plt.axvline(x = val.value, ls = '--')
    plt.title("spectrum in pixel form (Only Neon lines)")
    plt.xlabel(neon_spectra.spectral_axis.unit, fontsize=15)
    plt.ylabel(neon_spectra.flux.unit, fontsize=10)
    plt.show()
    
    #plt.xlim(0, 5000)
    #plt.ylim(0,1)
    return improved_xpts


def wave_cal( base_file= None, neon_spectra=None, standardize_wave_file=None, poly =3, autotol_value = 19, npixels=10):

    aprox_values = Table.read(base_file, names =('xpoints', 'waves'), format = 'ascii')
    neon_xpt = aprox_values['xpoints']
    neon_wpt= aprox_values['waves']
    
    #Now that we have our list of guessed x points and their related wavelength values,
    #we can use the following method to make sure our guessed x points are actually right on the peak of the wavelength.
    
    #improved_xpts = aprox_points(neon_spectra,neon_xpt, npixels)

    #while True:
     #   user = input('If you are happy with the approximation, enter "y" to continue or anything else to re-approximate the points: ')
      #  if user == 'y':
      #      break
      #  else:
       #     new_npixels = int(input('Enter a new value for npixels. It must be an integer:'))
        #    improved_xpts = aprox_points(neon_spectra,neon_xpt, new_npixels)
#Now that we have our xpts and our wpts, get them into the right format
#sci_wpts_ne = sci_wpts_ne * u.angstrom   #Make sure you only run this line once
    sci_xpts_ne_values = []
    for val in neon_xpt: #improved_xpts:
      sci_xpts_ne_values.append(val)#(val.value)
    
    print(sci_xpts_ne_values)
    print(neon_wpt)

    # sort, just in case
    srt = np.argsort(sci_xpts_ne_values)
    xpt = np.array(sci_xpts_ne_values)[srt]
    wpt = np.array(neon_wpt)[srt]
    #fpt = np.zeros_like(xpt)  # the fit wavelength points
    print(wpt)

    fit = np.polyfit(xpt, wpt, deg=poly)
    wavesolved = np.polyval(fit, neon_spectra.spectral_axis.value)
    
    srt = np.argsort(neon_xpt)
    xpt = np.array(neon_xpt)[srt]
    wpt = np.array(neon_wpt)[srt]
    fit = np.polyfit(xpt, wpt, deg=poly)
    wavesolved = np.polyval(fit, neon_spectra.spectral_axis.value)

    # a table of strong HeNeAr lines for the instrument in question, w/ units
    # apo_henear = Table.read(dir+'apohenear.ecsv', format='ascii.ecsv')['wave']
    #/content/kosmos_1/content/pykosmos/kosmos/resources/linelists/apohenear.dat

    # The old IRAF-style linelists just have 2 col: (wavelength, line name)
    henear_tbl = Table.read(standardize_wave_file,
                            names=('wave', 'name'), format='ascii')
    # IMPROVEMENT NEEDED: need to make `identify_nearest` point to these linelists itself?

    henear_tbl['wave'].unit = u.angstrom
    apo_henear = henear_tbl['wave']

    # Put in your value from the previous code.
    # Identify_nearest will take care of assing the calibration point automatically. You can play with autotol and see what fits your data best.
    sci_xpts, sci_wpts = identify_nearest(neon_spectra , wapprox= wavesolved*u.angstrom, linewave= apo_henear, autotol= autotol_value)

    if len(sci_wpts) != len(set(sci_wpts)):
        print('You have a duplicated wavelength point. Try changing autotol_value or redo identify_widget() with more points.')
    if len(sci_wpts) < 15:
      print('Your matched wavelength points is less than 15. If you have bad calibration points, you are advised to change autotol_value or redo identify_widget() with more points. for better calibration')

    return sci_xpts, sci_wpts