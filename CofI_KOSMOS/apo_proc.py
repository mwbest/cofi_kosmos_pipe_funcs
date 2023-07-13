def apo_proc(file, bias = None, flat = None, dark = None,
         trim = True, ilum = None, Saxis = 0, Waxis = 1,
         EXPTIME = 'EXPTIME', DATASEC = 'DATASEC',
         CR = False, GAIN = 'GAIN', READNOISE = 'RDNOISE', CRsigclip = 4.5):
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

    Saxis: int (optional), default is 0
    Set which axis is the spatial dimension. For DIS, Saxis=0 (corresponds to "NAXIS2" in the header).
    For KOSMOS, Saxis=1.

    Waxis: int (optional), default is 1
    Set which axis is the wavelength dimension. For DIS, Waxis=1 (corresponds to "NAXIS1" in the header).
    For KOSMOS, Waxis=0.
    Note: if Saxis is changed, Waxis will be updated, and visa versa.

    CR: bool, default = False
    If True, use the L.A. Cosmic routine to remove cosmic rays from the image before reducing.

    GAIN: string (optional), default = "GAIN"
    "Fits" header field containing the "GAIN" keyword used by L.A. Cosmic.

    READNOISE: string (optional), default = "RDNOISE"
    "Fits" header field containing the "RDNOISE" keyword used by L.A. Cosmic.

    CRsigclip: int (optional), default = 4.5
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
    # Old DIS default was Saxis=0, Waxis=1, shape = (1028,2048).
    # KOSMOS is swapped, shape = (4096, 2148).
    if (Saxis == 1) | (Waxis == 0):
        # If either axis is swapped, swap them both.
        Saxis = 1
        Waxis = 0
    # If desired, trimming to the illuminated region of the CCD.
    if ilum is None:
        pass
    else:
        if Waxis == 1:
            img = trim_image(img[ilum[0]:(ilum[-1] + 1), :])
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
