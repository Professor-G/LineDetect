.. _Examples:

Examples
===========
The examples below demonstrate how to use the `spectra_processor <https://linedetect.readthedocs.io/en/latest/autoapi/LineDetect/spectra_processor/index.html#LineDetect.spectra_processor.Spectrum>`_ class to generate the spectra continuum flux and identify specific abosrption features. 

1) Spectrum
-----------
To process your spectral data, initiliaze the ``Spectrum`` class and set the optional arguments. While explicitly designated below, the spectral ``line`` to search for is pre-set to 'MgII', with a default ``resolution_range`` of 1400 to 1700, corresponding to the minimum and maximum pixel resolution (in km/s). 

.. code-block:: python

    from LineDetect import spectra_processor

    spec = spectra_processor.Spectrum(line='MgII', resolution_range=(1400,1700))

Additional arguments are included to control the continuum generation, including the filtering ``method`` which defaults to a robust moving median, as well as the half-window size of the kernel, ``halfWindow``, and ``poly_order`` -- the order of the polynomial used to fit the flux (applicable for certain filters).

To process a single sample, call the ``process_spectrum`` method -- the arguments include the redshift, ``z``, of the object, the ``Lambda`` array of wavelengths, and the corresponding ``flux`` and ``flux_err`` arrays. Additionally, a ``qso_name`` can be input to differentiate between the saved entries, otherwise the order at which it was saved will be the sole identifier.

.. code-block:: python
    
    spec.process_spectrum(Lambda, flux, flux_err, z=z, qso_name='Obj_Name')

A DataFrame is saved as the ``df`` attribute, and if the specified spectral ``line`` is detected in the spectrum, then the data will be appended to the DataFrame. Note that by default the ``save_all`` class attribute is set to True, which will save entries for which there are no positive detections; these entries will contain the ``qso_name`` followed by 'None' values. If ``save_all`` is set to False, only spectra with positive ``line`` detection will be appended to the ``df`` attribute.

After running the ``process_spectrum`` method, the instantiated class will contain the ``continuum`` and ``continuum_err`` array attributes. These will be used automatically when calling the ``plot`` method:

.. code-block:: python

    spec.plot(include='both', errorbar=False, xlim=(4350,4385), ylim=(0.9,1.9),savefig=False)

The ``include`` parameter can be set to either 'spectrum' to plot the flux only, 'continuum' to display only the continuum fit, or 'both' for both options.

**IMPORTANT**: If no line is found it is possible that the continuum was insufficiently estimated as a result of low S/N, therefore it is avised to experiment with the different filtering options to identify the most appropriate algorithm for your dataset. To experiment with these parameters, change the ``method``, ``halfWindow``, and ``poly_order`` and either call the ``process_spectrum`` method again (which will overwrite the ``continuum`` and ``continuum_err`` attributes as per the new fit) or, if already called at least once, run the ``_reprocess`` method which requires no input as it calls the pre-loaded attributes.

.. code-block:: python
    
    spec.method = 'savgol' #Savitzky-Golay filter 
    spec.halfWindow, spec.poly_order = 100, 5

    spec._reprocess()

If no line is found a message will appear, if this is occursm the ``plot`` method can then be called again (with the updated continuum) to inspect the accuracy of the fit.

Note that currently only one line can be processed at a time, so to process multiple for a given set of data, we can run the methods consecutively after updating the attributes:

.. code-block:: python
    	
    #Set the first spectral line, note the unique qso_name
    spec.line = 'MgII'
    spec.process_spectrum(Lambda, flux, flux_err, z=z, qso_name='MgII_Obj_Name')

    #Set the second spectral line and _reprocess(), if qso_name is not updated it will re-use the name!
    spec.line = 'CaIV'
    spec._reprocess(qso_name='CaIV_Obj_Name') 

    #Set the third spectral line and _reprocess()..

2) Directory
-----------
As the DataFrame, ``df``, appends new results every time (if ``save_file`` is set to True), files from a directory can be processed at any point, although ccurrently the system supports only the fits format with the following header information:

**[0].header['Z'] is the redshift of the source, [0].data is the 1-D flux, and hdu[1].data the corresponding flux error.**

**[0].header must also contain the redshift information (float) and the appropriate coordinate conversion factor so as to invoke the Astropy World Coordinate System**

To load fits files from a directory, set the ``directory`` attribute and call the ``process_files`` method -- note that the ``qso_name`` that will be saved to the DataFrame will be automatically set to the file name.

.. code-block:: python
	
	spec.directory = '/Path/to/dir/'
	spec.process_files()    

	#Process another directory, the identified lines will be appended to the DataFrame
	spec.directory = '/Path/to/different/dir/'
	spec.process_files()

Unlike when processing single spectra with ``process_spectrum``, this method does not save ``continuum`` and ``continuum_err`` attributes, therefore the ``plot`` method cannot be called to view these samples, they will have to loaded individually for plotting purposes. 


