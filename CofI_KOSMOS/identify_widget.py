def identify_widget(arcspec, silent=False):
    """
    Interactive version of the Identify GUI, specifically using ipython widgets.

    Each line is roughly identified by the user, then a Gaussian is fit to
    determine the precise line center. The reference value for the line is then
    entered by the user.

    When finished, the output lines should usually be passed in a new Jupter
    notebook cell to `identify` for determining the wavelength solution:
    >>>> xpl,wav = identify_widget(arcspec) # doctest: +SKIP
    >>>> fit_spec = fit_wavelength(obj_spec, xpl, wav) # doctest: +SKIP

    NOTE: Because of the widgets used, this is not well suited for inclusion in
    pipelines, and instead is ideal for interactive analysis.

    Parameters
    ----------
    arcspec : Spectrum1D
        the 1d spectrum of the arc lamp to be fit.
    silent : bool, optional (default is False)
        Set to True to silence the instruction print out each time.

    Returns
    -------
    The pixel locations and wavelengths of the identified lines:
    pixel, wavelength
    """

    # the fluxes & pixels within the arc-spectrum
    flux = arcspec.flux.value
    xpixels = arcspec.spectral_axis.value

    msg = '''
    Instructions:
    ------------
    0) For proper interactive widgets, ensure you're using the Notebook backend
    in the Jupyter notebook, e.g.:
        %matplotlib notebook
    1) Click on arc-line features (peaks) in the plot. The Pixel Value box should update.
    2) Select the known wavelength of the feature in the Wavelength box.
    3) Click the Assign button, and a red line will be drawn marking the feature.
    4) When you've identified all your lines, stop the interaction for (or close) the figure.'''

    if not silent:
        print(msg)

    xpxl = []
    waves = []

    # Create widgets, two text boxes and a button
    xval = widgets.BoundedFloatText(
        value=5555.0,
        min=np.nanmin(xpixels),
        max=np.nanmax(xpixels),
        step=0.1,
        description='Pixel Value (from click):',
        style={'description_width': 'initial'})

           #asta
    linename =     widgets.Dropdown(
        options=['7032.41', '7245.17', '6402.25', '6506.53', '6929.47', '6143.06', '6678.28', 
        '6717.04', '6334.43', '7438.90', '8377.61', '8495.36', '8780.62', '8654.38', '8300.36'],
        #value='Enter Wavelength',
        description='Wavelength:',
        disabled=False,
      style={'description_width': 'initial'})

    button = widgets.Button(description='Assign')

    fig, ax = plt.subplots(figsize=(9, 3))

    # Handle plot clicks
    def onselect(eclick, erelease):
    # Get the region within the zoom box
    x1, x2 = sorted([eclick.xdata, erelease.xdata])
    rgn = np.where((xpixels >= x1) & (xpixels <= x2))[0]
    try:
        sig_guess = 3.
        p0 = [np.nanmax(flux[rgn]), np.nanmedian(flux), xpixels[np.argmax(flux[rgn])], sig_guess]
        popt, _ = curve_fit(_gaus, xpixels[rgn], flux[rgn], p0=p0)
        # Record x value of the highest peak in the text box
        xval.value = popt[2]
    except RuntimeError:
        # Fall back to the x value of the highest flux in the region if the fit doesn't work
        xval.value = xpixels[np.argmax(flux[rgn])]
    # Draw a vertical line at the x value of the highest peak
    ax.axvline(xval.value, lw=1, c='r', alpha=0.7)
    plt.draw()

    # Create a rectangle selector
    toggle_selector = RectangleSelector(ax, onselect, drawtype='box', useblit=True,
                                        button=[1],  # Left mouse button
                                        minspanx=5, minspany=5,
                                        spancoords='pixels',
                                        interactive=True)

    #fig.canvas.mpl_connect('button_press_event', onplotclick)

    # Handle button clicks
    def onbuttonclick(_):
        xpxl.append(xval.value)
        waves.append(float(linename.value))
        print(xpxl, waves)

        ax.axvline(xval.value, lw=1, c='r', alpha=0.7)
        return

    button.on_click(onbuttonclick)

    # Do the plot
    ax.plot(xpixels, flux)
    plt.draw()

    # Display widgets
    display(widgets.HBox([xval, linename, button]))

    # return np.array(xpxl), np.array(waves)
    return xpxl, waves

    #return the list of x pixel values and wavelength
    def get_results():
      return np.array(xpxl), np.array(waves)