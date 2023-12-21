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

__all__ = ['apo_proc', 'darkcombine']
def apo_proc(file, bias = None, flat = None, dark = None,
         trim = True, ilum = None, Saxis = 1, Waxis = 0,
         EXPTIME = 'EXPTIME', DATASEC = 'DATASEC',
         CR = True, GAIN = 0.6, READNOISE = 6, CRsigclip = 35):
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

    GAIN: string (optional), default = 0.6
    "Fits" header field containing the "GAIN" keyword used by L.A. Cosmic.

    READNOISE: string (optional), default = 6
    "Fits" header field containing the "RDNOISE" keyword used by L.A. Cosmic.

    CRsigclip: int (optional), default = 35
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
        img = cosmicray_lacosmic(img, gain = GAIN * u.electron / u.adu,
                                 readnoise = READNOISE * u.electron,
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
def darkcombine(darkfiles_list, bias, directory, trim = False, ilum_value = None):
    """
    This function first checks whether the dark files provided by user
    have the same exposure time. If not, it gives an error and interrupts execution.
    Then, it subtracts bias from each individual dark frame and uses Jim Davenport's
    "pykosmos.biascombine" function to combine the calibrated dark frames.
    Finally, it appends "EXPTIME" keyword to the header of the file containing
    the combined dark frame that has the same exposure time value as the raw
    dark files provided by user. Please read the "Parameters" for information
    regarding "trim" and "ilum_value" parameters.

    Parameters:

    darkfiles_list: numpy ndarray
    An array of dark files to combine.

    bias: string
    Path to the "fits" file containing the combined bias frame.

    directory: string
    Path to the place where each calibrated dark file
    as well as the resulting combined dark frame will be stored.

    trim: bool (optional), default is False
    If set to True, trim the sections that correspond to "bias" and "ilum"
    parameters in the "apo_proc" function also used to reduce a science image.
    This is only done so that a user can compare the combined dark frame and
    science image in ds9. This step makes the combined dark frame the same
    size as the reduced science image. If the user needs to use the combined dark frame later
    in the pipeline, he/she needs to use the dark frame that hasn't been trimmed.
    Trimming is ONLY for COMPARISON in DS9!!! This parameter can only be set to True
    once the user ran the function once, because "trim" requires another parameter
    that comes from "flatcombine" function below, and that function takes untrimmed
    combined dark frame as an optional parameter. If the user decides to plot the combined
    dark frame in ds9 against the science image, the user needs to come back to
    this function, set this parameter to True, and provide a value for the following parameter.

    ilum: numpy array (optional), default is None
    If trim is set to True, this value must be provided. This is the ilum value from
    the "flatcombine" function below.

    Returns:

    combined_darks: CCDData object or
    combined_darks_trimmed: CCDData object (if trim = True)

    """
    # This block of code checks whether the dark files provided by the user
    # have the same exposure time. If not, it returns an error and interrupts execution.
    exp_test = set()
    for darkfile in darkfiles_list:
      darksingle = fits.open(darkfile)
      header = darksingle[0].header
      exp = header['EXPTIME']
      exp_test.add(exp)
    assert len(exp_test) == 1, 'Not all darks have the same EXPTIME. distinct EXPTIMES in the files are:{}'.format(exp_test)
    # Making "directory" a path object
    path = Path(directory)
    # This block of code subtracts bias from each dark file, writes the calibrated
    # dark frames into text file "darkfiles.txt", and saves them in the directory
    # provided by the user.
    # Please make sure to run the following block of code (until the next comment) only once,
    # otherwise, you will be attempting to save files that already exist.
    # Thus, please comment this block of code out once you run it.
    with open('darkfiles.txt', 'w') as f:
      count = 1
      for darkfile in darkfiles_list:
        calibrated_darkfile = fits.getdata(darkfile) - fits.getdata(bias)
        f.write("calibrated_darkfile{}.fits".format(count) + "\n")
        filename = "calibrated_darkfile{}.fits".format(count)
        hdu = fits.PrimaryHDU(calibrated_darkfile)
        hdu.writeto(path/filename, overwrite = True)
        count += 1
    # Read the text file into a table, the list of column names to use has only one name, "impath".
    # Returning an array of the calibrated dark files.
    darkfiles_i = pd.read_table("darkfiles.txt", names = ['impath'])
    darkfiles = directory + darkfiles_i['impath'].values
    # Using Jim Davenport's "pykosmos.biascombine" function to combine calibrated dark frames.
    calibrated_combined_darkfiles = pykosmos.biascombine(darkfiles)
    filename = 'calibrated_combined_darkfiles.fits'
    combined_dark_path = directory + filename
    # The code below saves the combined dark frame as a "fits" file.
    # Please make sure to run the following block of code (until the next comment) only once,
    # otherwise, you will be attempting to save a file that already exists.
    # Thus, please comment this block of code out once you run it.
    hdu = fits.PrimaryHDU(calibrated_combined_darkfiles)
    hdu.writeto(path/ filename, overwrite = True)
    # The block below appends "EXPTIME" keyword to the header of the file containing the combined dark frame.
    # Appended "EXPTIME" has the same value as that of the raw dark frames provided by the user.
    # Please run the following block of code (until the next comment) only once,
    # otherwise the keyword will be appended to the header every time you run it.
    # To avoid running this code multiple times, after running it once, just comment it out.
    hdul = fits.open(combined_dark_path,'update')
    hdr = hdul[0].header
    hdr.append("EXPTIME")
    hdr["EXPTIME"] = exp
    hdul.writeto(combined_dark_path, overwrite = True)
    # If trim is set to True, trim the combined dark frame to the size of the science image and return trimmed
    # combined dark frame as a CCDData object. In order to do that, first the "DATASEC" keyword is appended.
    # Otherwise, just return the combined dark frame as a CCDData object.
    if trim == True:
      # Appending "DATASEC" keyword to the header of the combined dark frame.
      # The "DATASEC" combines into one "CSEC11" and "CSEC12" - the two data sections of CCD.
      # Please run the following block of code (until the next comment) only once,
      # otherwise the keyword will be appended to the header every time you run it.
      # To avoid running this code multiple times, after running it once, just comment it out.
      dark_frame = fits.open(combined_dark_path, 'update')
      hdr = dark_frame[0].header
      hdr.append("DATASEC")
      hdr["DATASEC"] = "[1:2048,1:4096]"
      hdr.comments["DATASEC"] = "data section of CCD (unbinned)"
      filename2 = "calibrated_combined_trimmed_darkfiles.fits"
      combined_dark_trimmed_path = directory + filename2
      dark_frame.writeto(combined_dark_trimmed_path, overwrite = True)
      # Trim the dark frame using "apo_proc" function.
      dark_frame_trimmed = apo_proc(combined_dark_trimmed_path, ilum = ilum_value, Saxis = 1, Waxis =0)
      # The code below saves the combined trimmed dark frame as a "fits" file.
      # Please make sure to run the following block of code (until the next comment) only once,
      # otherwise, you will be attempting to save a file that already exists.
      # Thus, please comment this block of code out once you run it.
      hdu = fits.PrimaryHDU(dark_frame_trimmed)
      hdu.writeto(path / filename2, overwrite = True)
      # Initialize the combined trimmed dark frame as a CCDData object and return it.
      combined_darks_trimmed = CCDData.read(combined_dark_trimmed_path, unit = 'adu')
      return combined_darks_trimmed
    else:
      # Initialize the combined dark frame as a CCDData object and return it.
      combined_darks = CCDData.read(combined_dark_path, unit = 'adu')
      return combined_darks
