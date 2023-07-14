#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 1 06:43:54 2023

@author: daniel
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from operator import itemgetter
import matplotlib.pyplot as plt  ## To plot the spectrum

from astropy.io import fits  ## To read the spectrum and load the wavelengths and flux into arrays
from astropy.wcs import WCS
from progress.bar import FillingSquaresBar

from LineDetect.continuum_finder import Continuum
from LineDetect.detect_elements import MgII 

class Spectrum:
    """
    A class for processing spectral data stored in FITS files.

    Can process either a set of .fits files or single spectra.

    Note:
        If the line is detected the spectrum features will be added
        to the DataFrame `df` attribute, which will always append new detections. 
        If no line is detected then nothing will be added to the DataFrame,
        but a message with the object name will print.

    Args:
        halfWindow (int, list, np.ndarray): The half-size of the window/kernel (in Angstroms) used to compute the continuum. 
            If this is a list/array of integers, then the continuum will be calculated
            as the median curve across the fits across all half-window sizes in the list/array.
            Defaults to 25.
        poly_order (int): The order of the polynomial used for smoothing the spectrum.
        resolution_range (tuple): A tuple of the minimum and maximum resolution (in km/s) used to detect MgII absorption.
            Can also be an integer or a float.
        directory (str): The path to the directory containing the FITS files. Defaults to None.
        save_all (bool): Parameter to control whether to save the non-detections. If the spectral feature is not detected 
            and save_all=True, the qso_name will be appended alongside 'None' entries. Defaults to False to save only positive detections.

    Methods:
        process_files(): Process the FITS files in the directory.
        process_spectrum(Lambda, y, sig_y, z, file_name): Process a single instance of spectral data.
        _reprocess(): Re-runs the process_spectrum method using the saved spectrum attributes.
        plot(include, errorbar, xlim, ylim, xlog, ylog, savefig): Plots the spectrum and/or continuum.
        find_MgII_absorption(Lambda, y, yC, sig_y, sig_yC, z, qso_name): Find the MgII lines, if present.
        find_CIV_absorption(Lambda, y, yC, sig_y, sig_yC, z, qso_name): Find the CIV lines, if present.
    """

    def __init__(self, halfWindow=25, resolution_range=(1400, 1700), 
        resolution_element=3, N_sig_1=5, N_sig_2=3, rest_wavelength_1=2796.35,
        rest_wavelength_2=2803.53, directory=None, save_all=False):
        
        self.halfWindow = halfWindow
        self.resolution_range = resolution_range
        self.resolution_element = resolution_element
        self.N_sig_1 = N_sig_1
        self.N_sig_2 = N_sig_2
        self.rest_wavelength_1 = rest_wavelength_1
        self.rest_wavelength_2 = rest_wavelength_2

        self.directory = directory
        self.save_all = save_all

        #Declare a dataframe to hold the info
        self.df = pd.DataFrame(columns=['QSO', 'Wavelength', 'z', 'W', 'deltaW']) 

    def process_files(self):
        """
        Processes each FITS file in the directory, detecting any Mg II absorption that may be present.

        The method iterates through each FITS file in the directory specified during initialization, 
        reads in the spectrum data and associated header information, applies continuum normalization, 
        identifies Mg II absorption features, and calculates the equivalent widths of said absorptions.
        The results are stored in a pandas DataFrame (df attribute). 
        
        Note:
            Unlike when processing single spectra, this method does not save
            the continuum and continuum_err attributes, therefore the plot()
            method cannot be called. Load a single spectrum using process_spectrum
            to save the continuum attributes.

        Returns:
            None
        """

        for i, (root, dirs, files) in enumerate(os.walk(os.path.abspath(self.directory))):
            progress_bar = FillingSquaresBar('Processing files......', max=len(files))
            for file in files:
                #Read each file in the directory
                try:
                    hdu = fits.open(os.path.join(root, file))
                except OSError:
                    print(); print('Invalid file type, skipping file: {}'.format(file))
                    progress_bar.next(); continue
                #Get the flux intensity and corresponding error array
                flux, flux_err = hdu[0].data, np.sqrt(hdu[1].data)
                #Recreate the wavelength spectrum from the info given in the WCS of the header
                w = WCS(hdu[0].header, naxis=1, relax=False, fix=False)
                Lambda = w.wcs_pix2world(np.arange(len(flux)), 0)[0]

                #Cut the spectrum blueward of the LyAlpha line
                z = hdu[0].header['Z'] #Redshift
                #Cut the spectrum blueward of the LyAlpha line
                Lya = (1 + z) * 1216 + 20 #Lya Line at 121.6 nm
                mask1 = (Lambda > Lya) 
                rest_frame = (1 + z) * rest_wavelength_1
                mask2 = (Lambda[mask1] < rest_frame)

                mask = mask1[mask2]
                Lambda, flux, flux_err = Lambda[mask], flux[mask], flux_err[mask]
                
                try:
                    #Generate the contiuum
                    continuum = Continuum(Lambda, flux, flux_err, halfWindow=self.halfWindow, N_sig=self.N_sig_2, resolution_element=self.resolution_element)
                    continuum.estimate()
                except ValueError: #This will catch the failed to fit message!
                    print(); print('Failed to fit the continuum, skipping file: {}'.format(file))
                    progress_bar.next(); continue
                #Find the MgII Absorption
                self.find_MgII_absorption(Lambda, flux, continuum.continuum, flux_err, continuum.continuum_err, z=z, qso_name=file)
                
                progress_bar.next()

        progress_bar.finish()

        return 

    def process_spectrum(self, Lambda, flux, flux_err, z, qso_name=None):
        """
        Processes a single spectrum, detecting any Mg II absorption that may be present.

        Args:
            Lambda (array-like): An array-like object containing the wavelength values of the spectrum.
            flux (array-like): An array-like object containing the flux values of the spectrum.
            flux_err (array-like): An array-like object containing the flux error values of the spectrum.
            z (float): The redshift of the QSO associated with the spectrum.
            qso_name (str, optional): The name of the QSO associated with the spectrum, will be
                saved in the DataFrame. Defaults to None, in which case 'No_Name' is used.

        Returns:
            None
        """

        qso_name = 'No_Name' if qso_name is None else qso_name

        #Cut the spectrum blueward of the LyAlpha line
        Lya = (1 + z) * 1216 + 20 #Lya Line at 121.6 nm
        rest_frame = (1 + z) * self.rest_wavelength_1
        mask = np.where((Lambda > Lya)&(Lambda < rest_frame))
        
        Lambda, flux, flux_err = Lambda[mask], flux[mask], flux_err[mask]
        
        #Generate the contiuum
        continuum = Continuum(Lambda, flux, flux_err, halfWindow=self.halfWindow, N_sig=self.N_sig_2, resolution_element=self.resolution_element)
        continuum.estimate()
        #Save the continuum attributes
        self.continuum, self.continuum_err = continuum.continuum, continuum.continuum_err

        #Find the MgII Absorption
        self.find_MgII_absorption(Lambda, flux, self.continuum, flux_err, self.continuum_err, z=z, qso_name=qso_name)
                
        self.Lambda, self.flux, self.flux_err, self.z, self.qso_name = Lambda, flux, flux_err, z, qso_name #For plotting

        return

    def _reprocess(self, qso_name=None):
        """
        Reprocesses the data, intended to be used after running process_spectrum().
        Useful for changing the attributes and quickly re-running the same sample.
        
        Note:
            This will update the DataFrame by appending the new object line features (if found).
        
        Args:
            qso_name (str, optional):

        Returns:
            None
        """

        qso_name = self.qso_name if qso_name is None else 'No_Name'

        #Cut the spectrum blueward of the LyAlpha line
        Lya = (1 + self.z) * 1216 + 20 #Lya Line at 121.6 nm
        mask = (self.Lambda > Lya) 
        self.Lambda, self.flux, self.flux_err = self.Lambda[mask], self.flux[mask], self.flux_err[mask]
  
        #Generate the contiuum
        continuum = Continuum(self.Lambda, self.flux, self.flux_err, halfWindow=self.halfWindow, N_sig=self.N_sig_2, resolution_element=self.resolution_element)
        continuum.estimate()
        #Save the continuum attributes
        self.continuum, self.continuum_err = continuum.continuum, continuum.continuum_err
        #Find the MgII Absorption
        self.find_MgII_absorption(self.Lambda, self.flux, self.continuum, self.flux_err, self.continuum_err, z=self.z, qso_name=self.qso_name)
        
        return 

    def plot(self, include='both', highlight=False, errorbar=False, xlim=None, ylim=None, xlog=False, ylog=False, 
        savefig=False, path=None):
        """
        Plots the spectrum and/or continuum.
    
        Args:
            include (float): Designates what to plot, options include
                'spectrum', 'continuum', or 'both.
            highlight (bool): If True then the line will be highlighted with accompanying
                vertical lines to visualize the equivalent width. Defaults to False.
            errorbar (bool): Whether to include the flux_err as y-errors. Defaults to False.
            xlim: Limits for the x-axis. Ex) xlim = (4000, 6000)
            ylim: Limits for the y-axis. Ex) ylim = (0.9, 0.94)
            xlog (boolean): If True the x-axis will be log-scaled.
                Defaults to True.
            ylog (boolean): If True the y-axis will be log-scaled.
                Defaults to False.
            savefig (bool): If True the figure will not disply but will be saved instead.
                Defaults to False.
            path (str, optional): Path in which the figure should be saved, defaults to None
                in which case the image is saved in the local home directory. 

        Returns:
            AxesImage
        """

        if self.continuum is None or self.flux is None:
            raise ValueError('This method only works after a single spectrum has been processed via the process_spectrum method.')

        if errorbar:
            continuum_err = self.continuum_err if include == 'continuum' or include == 'both' else None
            flux_err = self.flux_err if include == 'spectrum' or include == 'both' else None
        else:
            flux_err = continuum_err = None

        if include == 'continuum' or include == 'both':
            plt.errorbar(self.Lambda, self.continuum, yerr=continuum_err, fmt='r--', linewidth=0.6, label='Continuum')
        if include == 'spectrum' or include == 'both':
            plt.errorbar(self.Lambda, self.flux, yerr=flux_err, fmt='k-.', linewidth=0.2)
        
        plt.title(self.qso_name, size=14)
        plt.xlabel('Wavelength [Å]', size=12); plt.ylabel('Flux', alpha=1, color='k', size=12)
        plt.xticks(fontsize=10); plt.yticks(fontsize=12)
        plt.xscale('log') if xlog else None; plt.yscale('log') if ylog else None 
        plt.xlim(xlim) if xlim is not None else None; plt.ylim(ylim) if ylim is not None else None
        plt.legend(prop={'size': 12})#, loc='upper left')
        
        if highlight:
            if len(self.Mg2796) == 0:
                print('The highlight parameter is enabled but no line was detected!')
            else:
                for i in range(0, len(self.Mg2796) - 1, 2):
                    plt.axvline(x = self.Lambda[self.Mg2796[i]], color = 'orange')
                    plt.axvline(x = self.Lambda[self.Mg2796[i+1]], color = 'orange')
                    plt.axvline(x = self.Lambda[self.Mg2803[i]], color = 'red')
                    plt.axvline(x = self.Lambda[self.Mg2803[i + 1]], color = 'red')

        if savefig:
            path = str(Path.home()) if path is None else path 
            path += '/' if path[-1] != '/' else ''
            plt.savefig(path+'Spectrum_'+self.qso_name+'.png', dpi=300, bbox_inches='tight')
            print('Figure saved in: {}'.format(path)); plt.clf()
        else:
            plt.show()

        return 
        
    def find_MgII_absorption(self, Lambda, y, yC, sig_y, sig_yC, z, qso_name=None):
        """
        Finds Mg II absorption features in the QSO spectrum and adds the line information to the DataFrame,
        including the Equivalent Width and the corresponding error. 

        Args:
            Lambda (array-like): Wavelength array.
            y (array-like): Observed flux array.
            yC (array-like): Estimated continuum flux array.
            sig_y (array-like): Observed flux error array.
            sig_yC (array-like): Estimated continuum flux error array.
            z (float): The redshift of the QSO associated with the spectrum.
            qso_name (str, optional): The name of the QSO associated with the spectrum, will be
                saved in the DataFrame. Defaults to None, in which case 'No_Name' is used.
            
        Returns:
            None
        """

        #Declare an array to hold the resolution at each wavelength
        if isinstance(self.resolution_range, int) or isinstance(self.resolution_range, float):
            R = [[self.resolution_range] * len(Lambda)]
        else:
            R = np.linspace(self.resolution_range[0], self.resolution_range[1], len(Lambda))

        #The MgII function finds the lines
        Mg2796, Mg2803, EW2796, EW2803, deltaEW2796, deltaEW2803 = MgII(Lambda, y, yC, sig_y, sig_yC, R, N_sig_1=self.N_sig_1, N_sig_2=self.N_sig_2, 
            resolution_element=self.resolution_element, rest_wavelength_1=self.rest_wavelength_1, rest_wavelength_2=self.rest_wavelength_2)
        Mg2796, Mg2803 = np.array(Mg2796), np.array(Mg2803)
        self.Mg2796, self.Mg2803 = Mg2796.astype(int), Mg2803.astype(int)

        if len(self.Mg2796) != 0:
            for i in range(0, len(Mg2796) - 1, 2):
                wavelength = (Lambda[self.Mg2796[i]] + Lambda[self.Mg2796[i+1]])/2
                new_row = {'QSO': qso_name, 'Wavelength': wavelength, 'z': wavelength/self.rest_wavelength_1 - 1, 'W': EW2796[i], 'deltaW': deltaEW2796[i]}
                self.df = pd.concat([self.df, pd.DataFrame(new_row, index=[0])], ignore_index=True)
        else: 
            if self.save_all and 'EW' not in locals():
                new_row = {'QSO': qso_name, 'Wavelength': 'None', 'z': 'None', 'W': 'None', 'deltaW': 'None'}
                self.df = pd.concat([self.df, pd.DataFrame(new_row, index=[0])], ignore_index=True)
            
        return 
