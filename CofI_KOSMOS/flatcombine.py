# Making all necessary imports
import numpy as np
from pathlib import Path
from ccdproc import CCDData
from ccdproc import Combiner, trim_image
from ccdproc import cosmicray_lacosmic
import pandas as pd
from astropy.table import Table
from astropy import units as u
from astropy.io import fits
import sys
sys.path.append("/content/pykosmos")
import pykosmos

__all__ = ['apo_proc','flatcombine']
def apo_proc(file, bias = None, flat = None, dark = None,
         trim = True, ilum = None, Saxis = 1, Waxis = 0,
         EXPTIME = 'EXPTIME', DATASEC = 'DATASEC',
         CR = True, GAIN = 'GAIN', READNOISE = 'RDNOISE', CRsigclip = 0.01):
    """
    Semi-generalized function to read a "fits" file in, divide by exposure
    time (returns units of ADU/s), and optionally perform basic CCD
    processing to it (bias, dark, flat corrections, biassec and
    illumination region trimming).

    Parameters:

    file: string
    Path to the "fits" file.

    bias: CCDData object (optional), default = None
    Median bias frame to subtract from each flat image.

    dark: CCDData object (optional)
    Dark frame to subtract.

    flat: CCDData object (optional)
    Combined flat frame to divide.

    trim: bool, default = True
    Trim the "bias section" out of each flat frame. Uses "fits" header field defined by "DATASEC" keyword.

    ilum: array (optional)
    If provided, trim image to the illuminated portion of the CCD.

    EXPTIME: string (optional), default = "EXPTIME"
    "Fits" header field containing the exposure time in seconds

    DATASEC: string (optional), default = "DATASEC"
    "Fits" header field containing the data section of the CCD (to remove the bias section). Used if "trim=True".

    Saxis: int (optional), default is 1
    Set which axis is the spatial dimension. 

    Waxis: int (optional), default is 0
    Set which axis is the wavelength dimension. 

    CR: bool, default = True
    If True, use the L.A. Cosmic routine to remove cosmic rays from the image before reducing.

    GAIN: string (optional), default = "GAIN"
    "Fits" header field containing the "GAIN" keyword used by L.A. Cosmic.

    READNOISE: string (optional), default = "RDNOISE"
    "Fits" header field containing the "RDNOISE" keyword used by L.A. Cosmic.

    CRsigclip: int (optional), default = 0.01
    Sigma-clipping parameter passed to L.A. Cosmic.

    gain_apply: bool (optional), default = False
    Apply gain to image. If it is set to True and bias or flat are not in units of electrons per adu, this may result in an error.

    Returns:

    img: CCDData object

    """
    # Initializing the file provided by the user as a CCDData object.
    img = CCDData.read(file, unit = u.adu)

    # If cosmic rays are chosen to be removed, the following code will do so:
    if CR:
        img = cosmicray_lacosmic(img, gain = img.header[GAIN] * u.electron / u.adu,
                                 readnoise = img.header[READNOISE] * u.electron,
                                 sigclip = CRsigclip, gain_apply = False)

    # Subtracting bias (if it is provided by the user).
    if bias is None:
        pass
    else:
        img.data = img.data - bias
    # Scaling and subtracting dark (if it is provided by the user).
    if dark is None:
        pass
    else:
        scale = img.header['EXPTIME'] / dark.header['EXPTIME']
        img.data = img.data - dark.data * scale
    # If desired, trimming off the bias section.
    if trim:
        img = trim_image(img, fits_section = img.header[DATASEC])
    # If desired, trimming to the illuminated region of the CCD.
    if ilum is None:
        pass
    else:
        if Waxis == 0:
            img = trim_image(img[:, ilum[0]:(ilum[-1] + 1)])
    # Dividing out the flat (if it is provided by the user).
    if flat is None:
        pass
    else:
        img.data = img.data / flat
    img.data = img.data / img.header[EXPTIME]
    img.unit = img.unit / u.s
    # Returning the resulting image.
    return img
def flatcombine(ffiles, bias = None, dark = None, trim = True, normframe = True,
                illumcor = True, threshold = 0.9,
                responsecor = True, smooth = False, npix = 11,
                Saxis = 1, Waxis = 0,
                EXPTIME = 'EXPTIME', DATASEC = 'DATASEC'  # header keywords
                ):
    """
    A general-purpose wrapper function to create a science-ready
    flatfield image.

    Parameters:

    ffiles: numpy ndarray
    Array of paths to the flat frame "fits" files.

    bias: CCDData object (optional), default = None
    Median bias frame to subtract from each flat image.

    trim: bool, default = True
    Trim the "bias section" out of each flat frame. Uses "fits" header field defined by the "DATASEC" keyword.

    normframe: bool, default = True
    If set to True, normalize each bias frame by its median value before combining.

    illumcor: bool, default = True
    Use the median-combined flat to determine the illuminated portion of the CCD. Runs "find_illum" function.

    threshold: float (optional), default = 0.9
    Passed to "find_illum" function.The fraction to clip to determine the illuminated portion (between 0 and 1).

    responsecor: bool, default = True
    Divide out the spatially-averaged spectrum response from the flat image. Runs "flat_response" function.

    smooth: bool, default = False
    Passed to "flat_response" function. If desired, the 1D mean-combined flat is smoothed before dividing out.

    npix: int, default = 11
    Passed to "flat_response" function.
    If "smooth=True", determines how big of a boxcar smooth kernel should be used (in pixels).

    EXPTIME: string (optional), default = "EXPTIME"
    "Fits" header field containing the exposure time in seconds.

    DATASEC: string (optional), default = "DATASEC"
    "Fits" header field containing the data section of the CCD (to remove the bias section). Used if trim = True.

    Saxis: int (optional), default is 1
    Set which axis is the spatial dimension. 

    Waxis: int (optional), default is 0
    Set which axis is the wavelength dimension. 
    
    Returns:

    flat: CCDData object
    Always returned, the final flat image object.

    ilum: array
    Returned if illumcor = True. The 1D array to use for trimming science images to the illuminated portion of the CCD.

    """

    # Initialize an empty list "flist" to append reduced and normalized flat frames to it.
    flist = []
    # Loop over all flat frames.
    for ind in range(len(ffiles)):
        # Reduce each flat frame using "apo_proc" defined above.
        img = apo_proc(ffiles[ind], bias = bias, dark = dark, EXPTIME = EXPTIME, DATASEC = DATASEC, trim = trim)
        # If desired, normalize each flat frame by its median.
        if normframe:
            img.data = img.data / np.nanmedian(img.data)
        # Append each resulting flat frame to "flist".
        flist.append(img)
    # Combine the flat frames with a median.
    medflat = Combiner(flist).median_combine()
    # If desired, the median flat is used to detect the illuminated portion of the CCD.
    if illumcor:
        ilum = pykosmos.find_illum(medflat, threshold = threshold, Waxis = Waxis)
        # Trimming the median flat to only the illuminated portion.
        if Waxis == 0:
            medflat = trim_image(medflat[:, ilum[0]:(ilum[-1] + 1)])
    # If desired, divide out the spatially-averaged spectrum response from the flat image.
    if responsecor:
        medflat = pykosmos.flat_response(medflat, smooth = smooth, npix = npix, Saxis = Saxis)
    # If "illumcore" was set to True, return both the final flat image object
    # and the 1D array used for trimming science images to the illuminated portion of the CCD.
    if illumcor:
        return medflat, ilum
    else:
        return medflat
