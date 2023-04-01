from astropy.io import fits  ## To read the spectrum and load the wavelengths and flux into arrays
from astropy.wcs import WCS
import matplotlib.pyplot as plt  ## To plot the spectrum
import numpy as np
import pandas as pd
import os
from operator import itemgetter
from progress.bar import FillingSquaresBar

from Continuum import *
from detectElement import MgII


class SpectraProcessor:
    """
    A class for processing spectral data stored in FITS files.

    Args:
        directory (str): The path to the directory containing the FITS files.
        resolution_range (tuple): A tuple of the minimum and maximum resolution (in km/s) used to detect MgII absorption.
            Defaults to (1400, 1700).
        continuum_window (int): The size of the window (in Angstroms) used to compute the continuum. Defaults to 25.

    Attributes:
        directory (str): The path to the directory containing the FITS files.
        resolution_range (tuple): A tuple of the minimum and maximum resolution (in km/s) used to detect MgII absorption.
        continuum_window (int): The size of the window (in Angstroms) used to compute the continuum.
        df (pandas.DataFrame): A dataframe to store the results of the processing.

    Methods:
        process_files(): Process the FITS files in the directory and store the results in `df`.
        process_spectrum(Lambda, y, sig_y, z, file_name): Process a single instance of spectral data.
    """

    def __init__(self, directory, resolution_range=(1400, 1700), continuum_window=25):
        self.directory = directory
        self.resolution_range = resolution_range
        self.continuum_window = continuum_window
        self.df = pd.DataFrame(columns=['QSO', 'Wavelength', 'z', 'W', 'Delta W']) #Declare a dataframe to hold the info
     
    def process_fits_files(self):
        """
        Processes each FITS file in the directory, detecting any Mg II absorption that may be present.

        The method iterates through each FITS file in the directory specified during initialization, 
        reads in the spectrum data and associated header information, applies continuum normalization, 
        identifies Mg II absorption features, and calculates the equivalent widths of said absorptions.
        The results are stored in a pandas DataFrame (df attribute). 

        Returns:
            None
        """

        for i, (root, dirs, files) in enumerate(os.walk(os.path.abspath(self.directory))):
            progess_bar = bar.FillingSquaresBar('Processing files......', max=len(files))
            for file in files:
                #Read each file in the directory
                hdu = fits.open(os.path.join(root, file))

                #Get the flux intensity and corresponding error array
                y, sig_y= hdu[0].data, np.sqrt(hdu[1].data)
                #Recreate the wavelength spectrum from the info given in the WCS of the header
                w = WCS(hdu[0].header, naxis=1, relax=False, fix=False)
                Lambda = w.wcs_pix2world(np.arange(len(y)), 0)[0]

                #Cut the spectrum blueward of the LyAlpha line
                z = hdu[0].header['Z'] #Redshift
                Lya = (1+z) * 1216 + 20
                indexList = np.where(Lambda > Lya)[0]
                Lambda, y, sig_y = Lambda[indexList], y[indexList], sig_y[indexList]

                #Generate the contiuum
                yC, sig_yC = medianContinuum(Lambda, y, self.continuum_window)
                yC = legendreContinuum(Lambda, y, yC, sig_y, sig_yC)

                #Find the MgII Absorption
                self.find_MgII_absorption(Lambda, y, yC, sig_y, sig_yC)

                progess_bar.next()

        progess_bar.finish()

        return 

    def process_spectrum(self, Lambda, y, sig_y, redshift, qso_name=None):
        """
        Processes a single spectrum, detecting any Mg II absorption that may be present.

        Args:
            Lambda (array-like): An array-like object containing the wavelength values of the spectrum.
            y (array-like): An array-like object containing the flux values of the spectrum.
            sig_y (array-like): An array-like object containing the flux error values of the spectrum.
            redshift (float): The redshift of the QSO associated with the spectrum.
            qso_name (str, optional): The name of the QSO associated with the spectrum, will be
                saved in the DataFrame. Defaults to None, in which case 'No_Name' is used.

        Returns:
            None
        """

        qso_name = 'No_Name' if qso_name is None else qso_name

        # Cut the spectrum blueward of the LyAlpha line
        Lya = (1 + redshift) * 1216 + 20
        indexList = np.where(Lambda > Lya)[0]
        Lambda, y, sig_y = Lambda[indexList], y[indexList], sig_y[indexList]

        # Declare an array to hold the resolution at each wavelength
        R = np.linspace(self.resolution_range[0], self.resolution_range[1], len(Lambda))

        # Generate the continuum
        yC, sig_yC = medianContinuum(Lambda, y, self.continuum_window)
        yC = legendreContinuum(Lambda, y, yC, sig_y, sig_yC)

        # Find the MgII Absorption
        self.find_MgII_absorption(Lambda, y, yC, sig_y, sig_yC, R)

        return

    def find_MgII_absorption(self, Lambda, y, yC, sig_y, sig_yC, R):
        """
        Finds Mg II absorption features in the QSO spectrum and adds the results to the DataFrame.

        Args:
            Lambda (array-like): Wavelength array.
            y (array-like): Observed flux array.
            yC (array-like): Estimated continuum flux array.
            sig_y (array-like): Observed flux error array.
            sig_yC (array-like): Estimated continuum flux error array.
            R (float): Rest-frame equivalent width ratio, i.e., EW_2796 / EW_2803, used for continuum normalization.

        Returns:
            None
        """

        #Declare an array to hold the resolution at each wavelength
        R = np.linspace(self.resolution_range[0], self.resolution_range[1], len(Lambda))

        #The MgII function finds the lines
        Mg2796, Mg2803, EW2796, EW2803, deltaEW2796, deltaEW2803 = MgII(Lambda, y, yC, sig_y, sig_yC, R)
        Mg2796, Mg2803 = Mg2796.astype(int), Mg2803.astype(int)

        for i, Mg2796_i in enumerate(Mg2796[:-1:2]):
            Mg2803_i, Mg2796_j = Mg2803[2 * i], Mg2796[2 * i + 1]
            if Mg2803_i != Mg2796_j:
                Mg2796_slice = Mg2796[2 * i:2 * i + 2]
                wavelength = np.mean(Lambda[Mg2796_slice])
                EW, sigEW = apertureEW(Mg2796_slice[0], Mg2796_slice[-1], Lambda, y, yC, sig_y, sig_yC)
                row_data = [qso_name, wavelength, wavelength / 2796 - 1, EW, sigEW]
                self.df = self.df.append(pd.Series(row_data, index=self.df.columns), ignore_index=True)

        return 

