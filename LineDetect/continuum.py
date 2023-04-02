#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 1 10:14:10 2023

@author: daniel
"""
import numpy as np
from typing import Tuple
from ctypes import sizeof
from scipy.special import eval_legendre, betainc 
from scipy.signal import savgol_filter, butter, filtfilt
from scipy.ndimage.filters import gaussian_filter1d

from LineDetect.feature_finder import *

class Continuum:
    def __init__(self, Lambda, flux, flux_err, method='median', halfWindow=25, poly_order=2):
        """
        Generates the continuum and continuun error.
        
        Note:
            This module contains functions to fit a continuum across the quasar spectrum.
            There are three primary functions: 
                - median across a running window, 
                - linear interplation across absorption/emission gaps, and
                - Legendre polynomial fitting across absorption/emission gaps.

        The routines increase in accuracy from the top to bottom i.e. median method to Legendre fitting but have to be applied sequentially.
        This means to apply the Legendre fitting, both methods that precede it have to be applied to the spectrum. 

        Args:
            Lambda (np.ndarray): Array of wavelengths.
            flux (np.ndarray): Array of self.flux values.
            flux_err (np.ndarray): Array of uncertainties in the self.flux values.
            method (str): The method to apply when estimating the continuum, options include: 'median', 'savgol',
                'gaussian', or 'butter'. Defaults to 'median'
            halfWindow (int): The half-window size to use for the smoothing procedure.
                If this is a list/array of integers, then the continuum will be calculated
                as the median curve across the fits across all half-window sizes in the list/array.
                Defaults to 25.
            poly_order (int): The order of the polynomial used for smoothing the spectrum.
        """

        self.Lambda = Lambda
        self.flux = flux
        self.flux_err = flux_err
        self.method = method
        self.halfWindow = halfWindow
        self.poly_order = poly_order

        self.size = len(self.Lambda)

    def estimate(self, fit_legendre=True):
        """
        Args:
            fit_legendre (bool):
        """

        if self.method == 'median':
            self.median_filter()
        elif self.method == 'savgol':
            self.savgol_filter()
        elif self.method == 'butter':
            self.butterworth_filter()
        elif self.method == 'gaussian':
            self.gaussian_filter()
        else:
            raise ValueError("Invalid method, options are: 'median',  'savgol', 'butter', or 'gaussian'.")

        if fit_legendre:
            yC = self.legendreContinuum(region_size=150, max_order=20, p_threshold=0.05)
        
    def median_filter(self):
        """
        Smooths out the flux array with a robust median-based filter

        Returns:
            Creates the continuum and continuum_err attributes.
        """
        self.flux[~np.isfinite(self.flux)] = 0 #Replace NaN values with 0
        if isinstance(self.halfWindow, int):
            yC, sig_yC = np.zeros(self.size), np.zeros(self.size)
            # Loop to scan through each pixel in the spectrum
            for i in range(0, self.size):
                # Condition for the blue region of the spectrum where there is insufficient data to the left of the pixel
                if i < self.halfWindow:
                    yC[i] = np.nanmedian(self.flux[0: i + self.halfWindow])
                    sig_yC[i] = np.nanstd(self.flux[0: i + self.halfWindow])
                    continue

                # Condition for the red region of the spectrum where there is insufficient data to the right of the pixel
                if i > self.size - self.halfWindow:
                    yC[i] = np.nanmedian(self.flux[i - self.halfWindow: self.size])
                    sig_yC[i] = np.nanstd(self.flux[i - self.halfWindow: self.size])
                    continue

                # Not the end cases
                yC[i] = np.nanmedian(self.flux[i - self.halfWindow: i + self.halfWindow])
                sig_yC[i] = np.nanstd(self.flux[i - self.halfWindow: i + self.halfWindow])
            
            self.continuum, self.continuum_err = yC, sig_yC

        else:
            yC_list, sig_yC_list = [], []
            for window_size in self.halfWindow:
                yC, sig_yC = np.zeros(self.size), np.zeros(self.size)
                # Loop to scan through each pixel in the spectrum
                for i in range(0, self.size):
                    # Condition for the blue region of the spectrum where there is insufficient data to the left of the pixel
                    if i < window_size:
                        yC[i] = np.nanmedian(self.flux[0: i + window_size])
                        sig_yC[i] = np.nanstd(self.flux[0: i + window_size])
                        continue

                    # Condition for the red region of the spectrum where there is insufficient data to the right of the pixel
                    if i > self.size - window_size:
                        yC[i] = np.nanmedian(self.flux[i - window_size: self.size])
                        sig_yC[i] = np.nanstd(self.flux[i - window_size: self.size])
                        continue

                    # Not the end cases
                    yC[i] = np.nanmedian(self.flux[i - window_size: i + window_size])
                    sig_yC[i] = np.nanstd(self.flux[i - window_size: i + window_size])
                
                yC_list.append(yC); sig_yC_list.append(sig_yC)
            #Average across individual pixels
            yC, sig_yC = np.median(yC_list, axis=0), np.median(sig_yC_list, axis=0)

        self.continuum, self.continuum_err = yC, sig_yC

        return

    def savgol_filter(self):
        """
        Smooths out the flux array with a Savitzky-Golay filter.

        Non-finite values in the flux array will be set to zero as this 
        routine employs the scipy implementation of this filter
        which can't handle NaN entries.

        Returns:
            Creates the continuum and continuum_err attributes.
        """

        self.flux[~np.isfinite(self.flux)] = 0 #Replace NaN values with 0
        if isinstance(self.halfWindow, int):
            #Apply the filter
            yC = savgol_filter(self.flux, window_length=2 * self.halfWindow + 1, polyorder=self.poly_order, mode='mirror')
            sig_yC = np.zeros_like(self.flux)

            # Calculate the standard deviation using non-NaN values
            non_nan_mask = ~np.isnan(self.flux)

            #Generate the continuum error array
            for i in range(0, len(self.Lambda)):
                if i < self.halfWindow:
                    sig_yC[i] = np.nanstd(self.flux[0: i + self.halfWindow][non_nan_mask[0: i + self.halfWindow]] - yC[0: i + self.halfWindow][non_nan_mask[0: i + self.halfWindow]])
                    continue

                if i > len(self.Lambda) - self.halfWindow:
                    sig_yC[i] = np.nanstd(self.flux[i - self.halfWindow: self.size][non_nan_mask[i - self.halfWindow: self.size]] - yC[i - self.halfWindow: self.size][non_nan_mask[i - self.halfWindow: self.size]])
                    continue

                sig_yC[i] = np.nanstd(self.flux[i - self.halfWindow: i + self.halfWindow][non_nan_mask[i - self.halfWindow: i + self.halfWindow]] - yC[i - self.halfWindow: i + self.halfWindow][non_nan_mask[i - self.halfWindow: i + self.halfWindow]])
        
        else:        
            yC_list, sig_yC_list = [], []
            for window_size in self.halfWindow:
                #Apply filter
                yC = savgol_filter(self.flux, window_length=2 * window_size + 1, polyorder=self.poly_order, mode='mirror')
                yC_list.append(yC)
                #This will be the error array for this particular window_size
                sig_yC = np.zeros_like(self.flux)

                # Calculate the standard deviation using non-NaN values
                non_nan_mask = ~np.isnan(self.flux)

                #Generate the continuum error array
                for i in range(0, len(self.Lambda)):
                    if i < window_size:
                        sig_yC[i] = np.nanstd(self.flux[0: i + window_size][non_nan_mask[0: i + window_size]] - yC[0: i + window_size][non_nan_mask[0: i + window_size]])
                        continue

                    if i > len(self.Lambda) - window_size:
                        sig_yC[i] = np.nanstd(self.flux[i - window_size: self.size][non_nan_mask[i - window_size: self.size]] - yC[i - window_size: self.size][non_nan_mask[i - window_size: self.size]])
                        continue

                    sig_yC[i] = np.nanstd(self.flux[i - window_size: i + window_size][non_nan_mask[i - window_size: i + window_size]] - yC[i - window_size: i + window_size][non_nan_mask[i - window_size: i + window_size]])
                sig_yC_list.append(sig_yC)
            #Average across individual pixels
            yC, sig_yC = np.median(yC_list, axis=0), np.median(sig_yC_list, axis=0)

        self.continuum, self.continuum_err = yC, sig_yC

        return

    def gaussian_filter(self, truncate=1):
        """
        Smooths out the flux array with a 1D Gaussian kernel.

        Non-finite values in the flux array will be set to zero as this 
        routine employs the scipy implementation of this filter
        which can't handle NaN entries.

        Args:
            truncate (int): Determines the range of values over which the Gaussian 
                function is calculated, and affects the smoothness and sharpness of 
                the resulting filter. A higher truncate yields a smoother fit. Defaults to 4.

        Returns:
            Creates the continuum and continuum_err attributes.
        """

        self.flux[~np.isfinite(self.flux)] = 0 #Replace NaN values with 0
        if isinstance(self.halfWindow, int):
            yC = gaussian_filter1d(self.flux, sigma=self.halfWindow, order=self.poly_order, mode='mirror', truncate=truncate)
            yC *= sum(self.flux)/sum(yC) #Otherwise the fit will be normalized and thus be way off from the actual spectrum! 
            sig_yC = np.zeros_like(self.flux)
            # Calculate the standard deviation using non-NaN values
            non_nan_mask = ~np.isnan(self.flux)

            for i in range(0, len(self.Lambda)):
                if i < self.halfWindow:
                    sig_yC[i] = np.nanstd(self.flux[0: i + self.halfWindow][non_nan_mask[0: i + self.halfWindow]] - yC[0: i + self.halfWindow][non_nan_mask[0: i + self.halfWindow]])
                    continue

                if i > len(self.Lambda) - self.halfWindow:
                    sig_yC[i] = np.nanstd(self.flux[i - self.halfWindow: self.size][non_nan_mask[i - self.halfWindow: self.size]] - yC[i - self.halfWindow: self.size][non_nan_mask[i - self.halfWindow: self.size]])
                    continue

                sig_yC[i] = np.nanstd(self.flux[i - self.halfWindow: i + self.halfWindow][non_nan_mask[i - self.halfWindow: i + self.halfWindow]] - yC[i - self.halfWindow: i + self.halfWindow][non_nan_mask[i - self.halfWindow: i + self.halfWindow]])

        else:        
            yC_list, sig_yC_list = [], []
            for window_size in self.halfWindow:
                yC = gaussian_filter1d(self.flux, sigma=window_size, order=self.poly_order, mode='mirror')
                yC_list.append(yC)
                #This will be the error array for this particular window_size
                sig_yC = np.zeros_like(self.flux)
                # Calculate the standard deviation using non-NaN values
                non_nan_mask = ~np.isnan(self.flux)

                for i in range(0, len(self.Lambda)):
                    if i < window_size:
                        sig_yC[i] = np.nanstd(self.flux[0: i + window_size][non_nan_mask[0: i + window_size]] - yC[0: i + window_size][non_nan_mask[0: i + window_size]])
                        continue

                    if i > len(self.Lambda) - window_size:
                        sig_yC[i] = np.nanstd(self.flux[i - window_size: self.size][non_nan_mask[i - window_size: self.size]] - yC[i - window_size: self.size][non_nan_mask[i - window_size: self.size]])
                        continue

                    sig_yC[i] = np.nanstd(self.flux[i - window_size: i + window_size][non_nan_mask[i - window_size: i + window_size]] - yC[i - window_size: i + window_size][non_nan_mask[i - window_size: i + window_size]])
                sig_yC_list.append(sig_yC)
            #Average across individual pixels
            yC, sig_yC = np.median(yC_list, axis=0), np.median(sig_yC_list, axis=0)

        self.continuum, self.continuum_err = yC, sig_yC

        return

    def butterworth_filter(self, sampling_rate=50):
        """
        Smooths out the flux array with a Butterworth filter.
    
        Note:
            If the observing instrument has a spectral resolution of 1 Angstrom, 
            for example, then setting the sampling_rate to 3 Angstrom/pixel would 
            be reasonable. In principle, setting this argument to be higher value
            will more accurately capture the continuum, at the expense of computational power.
            
            Note that the Nyquist frequency is defined as half of the sampling rate, 
            therefore the sampling rate should be at least twice the highest frequency 
            we wish to preserve. For example, given a resolution of sampling_rate=3 Angstrom/pixel,
            and a halfWindow of 25 pixels, the sampling_rate should be least 50 pixels to preserve 
            frequencies up to 1.5 pixels.

        Returns:
            Creates the continuum and continuum_err attributes.
        """

        self.flux[~np.isfinite(self.flux)] = 0 #Replace NaN values with 0
        # Define the filter parameters
        nyquist = 0.5 * sampling_rate

        if isinstance(self.halfWindow, int):
            #Set up the frequency implications
            cutoff = self.halfWindow / nyquist
            cutoff_norm = cutoff / nyquist  # Normalized cutoff frequency to avoid error that the value is not between 0 and 1
            b, a = butter(self.poly_order, cutoff_norm, 'lowpass')
            
            #Apply the filter
            yC = filtfilt(b, a, self.flux)
            sig_yC = np.zeros_like(self.flux)

            # Calculate the standard deviation using non-NaN values
            non_nan_mask = ~np.isnan(self.flux)

            #Generate the continuum error array
            for i in range(0, len(self.Lambda)):
                if i < self.halfWindow:
                    sig_yC[i] = np.nanstd(self.flux[0: i + self.halfWindow][non_nan_mask[0: i + self.halfWindow]] - yC[0: i + self.halfWindow][non_nan_mask[0: i + self.halfWindow]])
                    continue

                if i > len(self.Lambda) - self.halfWindow:
                    sig_yC[i] = np.nanstd(self.flux[i - self.halfWindow: self.size][non_nan_mask[i - self.halfWindow: self.size]] - yC[i - self.halfWindow: self.size][non_nan_mask[i - self.halfWindow: self.size]])
                    continue

                sig_yC[i] = np.nanstd(self.flux[i - self.halfWindow: i + self.halfWindow][non_nan_mask[i - self.halfWindow: i + self.halfWindow]] - yC[i - self.halfWindow: i + self.halfWindow][non_nan_mask[i - self.halfWindow: i + self.halfWindow]])

        else:        
            yC_list, sig_yC_list = [], []
            for window_size in self.halfWindow:
                #Set up the frequency implications
                cutoff = self.halfWindow / nyquist
                cutoff_norm = cutoff / nyquist  # Normalized cutoff frequency to avoid error that the value is not between 0 and 1
                b, a = butter(self.poly_order, cutoff_norm, 'lowpass')
                
                #Apply the filter and append array to list
                yC = filtfilt(b, a, self.flux)
                yC_list.append(yC)

                sig_yC = np.zeros_like(self.flux)

                # Calculate the standard deviation using non-NaN values
                non_nan_mask = ~np.isnan(self.flux)

                #Generate the continuum error array
                for i in range(0, len(self.Lambda)):
                    if i < window_size:
                        sig_yC[i] = np.nanstd(self.flux[0: i + window_size][non_nan_mask[0: i + window_size]] - yC[0: i + window_size][non_nan_mask[0: i + window_size]])
                        continue

                    if i > len(self.Lambda) - window_size:
                        sig_yC[i] = np.nanstd(self.flux[i - window_size: self.size][non_nan_mask[i - window_size: self.size]] - yC[i - window_size: self.size][non_nan_mask[i - window_size: self.size]])
                        continue

                    sig_yC[i] = np.nanstd(self.flux[i - window_size: i + window_size][non_nan_mask[i - window_size: i + window_size]] - yC[i - window_size: i + window_size][non_nan_mask[i - window_size: i + window_size]])
                sig_yC_list.append(sig_yC)
            #Average across individual pixels
            yC, sig_yC = np.median(yC_list, axis=0), np.median(sig_yC_list, axis=0)

        self.continuum, self.continuum_err = yC, sig_yC

        return

    def legendreContinuum(self, region_size=150, max_order=20, p_threshold=0.05):
        """
        Fits a Legendre polynomial to a spectrum to locate absorption and/or emission features.

        The function identifies absorption or emission features in the input spectrum using the `featureFinder` function,
        then extends thefl spectrum on both sides to avoid running out of bounds. For each feature, it selects a region
        of size region_size pixels around it and fits a Legendre polynomial to this region. If there are two features less than region_size pixels
        apart, the function computes the gap between them and includes the pixels between the features in the fitting array.
        The function returns an array of the fitted continuum values.

        Args:
            region_size (int): The size of the region to apply the polynomial fitting. Defaults to 150 pixels.
            max_order (int): The maximum order of the Legendre polynomial to fit. Defaults to 20.
            p_threshold (float): The p-value threshold for the F-test. If the p-value is greater than this threshold,
                the fit is considered acceptable and the continuum level is returned. Defaults to 0.05.

        Returns:
            A 1D array of the fitted continuum values for the input spectrum.
        """

        # Find the regions of absorption
        featureRange = featureFinder(self.Lambda, self.flux, self.continuum, self.flux_err, self.continuum_err)

        # Extend the spectrum on both sides by 150 pixels to avoid the risk of the code running out of bounds in the spectrum
        leftExtension = np.arange(np.floor(self.Lambda[0]) - region_size, np.floor(self.Lambda[0]))
        rightExtension = np.arange(np.ceil(self.Lambda[-1]), np.ceil(self.Lambda[-1]) + region_size)
        extendedLambda = np.concatenate((leftExtension, self.Lambda, rightExtension))
        extendedFlux = np.pad(self.flux, (region_size, region_size), mode='edge')
        extendedyC = np.pad(self.continuum, (region_size, region_size), mode='edge')
        extendedsig_y = np.pad(self.flux_err, (region_size, region_size), mode='edge')
        extendedsig_yC = np.pad(self.continuum_err, (region_size, region_size), mode='edge')

        i = 0 #Iterate through the list of absorption features and start fitting 
        #Define an array to hold the regions within 
        while i < len(featureRange):
            #Because of how the function is defined, there will always be 150 clean pixels to the left of the feature being considered.
            #This is because we ensure the preceding features are clear to the right 
            fitLeft = extendedLambda[featureRange[i] + 150 - region_size : featureRange[i] + 150]
            fitFlux = extendedFlux[featureRange[i] - region_size + 150: featureRange[i] + 150]
            fitsig_y = extendedsig_y[featureRange[i] - region_size + 150: featureRange[i] + 150]
            left = featureRange[i] - region_size + 150

            fitRight = []

            #If there are two absorption features within 150 pixels of each other, 
            # - compute the gap between the right of the first feature and the left of the second feature
            # - pick pixels to the right of the second feature till we hit 150 clean pixels

            #Define the number of pixels to be fit
            pixels = region_size

            #If there are two features not separated by 150 pixels,...
            if i < len(featureRange) - 2 and featureRange[i + 2] - featureRange[i + 1] < region_size:
                #Compute the separation of the features
                sep = featureRange[i + 2] - featureRange[i + 1]
                #Subtract this from 150
                pixels -= sep
                #Add the pixels between the two features to the fitting array
                fitRight = np.append(fitRight, extendedLambda[featureRange[i+1] + 150: featureRange[i+1] + 150 + sep])
                fitFlux = np.append(fitFlux, extendedFlux[featureRange[i+1] + 150: featureRange[i+1] + sep + 150])
                fitsig_y = np.append(fitsig_y, extendedsig_y[featureRange[i+1] + 150: featureRange[i+1] + sep + 150])

                #Start from the right of the second feature
                k = i + 3
                while pixels > 0:
                    
                    #If there are enough clean pixels present to the left of the current feature, we have our range
                    if k+1 == len(featureRange) or featureRange[k + 1] - featureRange[k] > pixels:
                        fitRight = np.append(fitRight, extendedLambda[featureRange[k] + 150: featureRange[k] + pixels + 150])
                        fitFlux = np.append(fitFlux, extendedFlux[featureRange[k] + 150: featureRange[k] + pixels + 150])
                        fitsig_y = np.append(fitsig_y, extendedsig_y[featureRange[k] + 150: featureRange[k] + pixels + 150])
                        #Update i to start from the left of the current pixel.
                        # Because we increment i by 2 in the outer loop, we will be at the start of the next feature at the end of this iteration. 
                        i = k - 1
                        right = featureRange[k] + pixels + 150
                        break
                    
                    #If there aren't enough pixels, save the pixels that are clean and repeat the process
                    sep = featureRange[k + 1] - featureRange[k]
                    pixels -= sep
                    fitRight = np.append(fitRight, extendedLambda[featureRange[k] + 150: featureRange[k] + 150 + sep])
                    fitFlux = np.append(fitFlux, extendedFlux[featureRange[k] + 150: featureRange[k] + 150 + sep])
                    fitsig_y = np.append(fitsig_y, extendedsig_y[featureRange[k] + 150: featureRange[k] + sep + 150])
                    right = featureRange[k] + 150 + sep
                    #Increment k by 2 because we only need to look the right of a feature
                    k += 2

            #If the spectrum has region_size clean pixels to the right, no problems
            else:
                fitRight = np.append(fitRight, extendedLambda[featureRange[i+1] + 150: featureRange[i+1] + 150 + region_size])
                fitFlux = np.append(fitFlux, extendedFlux[featureRange[i+1] + 150: featureRange[i+1] + 150 + region_size])
                fitsig_y = np.append(fitsig_y, extendedsig_y[featureRange[i+1] + 150: featureRange[i+1] + 150 +region_size])
                right = featureRange[i] + region_size + 150

            #Join the left and right ends of the wavelength array
            fitLambda = np.append(fitLeft, fitRight)

            #Find the functional fit of the continuum in this range
            tempLegendre = legendreFit(fitLambda, extendedLambda, fitFlux, fitsig_y, left, right, region_size, max_order=max_order, p_threshold=p_threshold)
            if np.any(tempLegendre):
                extendedyC[left: right + 1] = tempLegendre

            #Increment the iterator by 2 since there are two values associated with one feature. 
            i += 2

        self.continuum = extendedyC[150: len(extendedyC) - 150]

        return 


def legendreFit(fitLambda, extendedLambda, fitFlux, fitsigFlux, left, right, region_size, max_order=20, p_threshold=0.05):
    """
    Fits a Legendre polynomial to a given spectrum so as to estimate the continuum.

    Args:
        fitLambda (np.ndarray): The x-values of the spectrum to be fit.
        extendedLambda (np.ndarray): The x-values of the extended spectrum.
        fitFlux (np.ndarray): The y-values of the spectrum to be fit.
        fitsigFlux (np.ndarray): The uncertainties in the y-values.
        left (int): The index of the left-most point of the region to be fit.
        right (int): The index of the right-most point of the region to be fit.
        region_size (int): The number of points on either side of a given point to use in the fit.
        max_order (int): The maximum order of the Legendre polynomial to fit. Defaults to 20.
        p_threshold (float): The p-value threshold for the F-test. If the p-value is greater than this threshold,
            the fit is considered acceptable and the continuum level is returned. Defaults to 0.05.

    Returns:
        The continuum level of the spectrum.
    """

    #Convert the x-axis to the domain [-1, 1]
    Lambda_L, Lambda_U = fitLambda[0], fitLambda[-1]
    fitLambda_norm = (2 * fitLambda - Lambda_L - Lambda_U) / (Lambda_U - Lambda_L)

    ##Start at the first order Legendre Polynomial and compare it with the 0th order, and so on.
    for n in range(1, max_order + 1):
        #Fit the window with Legendre polynomial of degree n - 1 = m 
        fit_m = np.polynomial.legendre.legfit(fitLambda_norm, fitFlux, n - 1)

        #Fit the window with Legendre polynomial of degree n
        fit_n = np.polynomial.legendre.legfit(fitLambda_norm, fitFlux, n)
        
        #Find chi square for both the fits
        chiSq_m = legendreChiSq(fit_m, fitLambda_norm, fitFlux, fitsigFlux)
        chiSq_n = legendreChiSq(fit_n, fitLambda_norm, fitFlux, fitsigFlux)

        df1 = 2 * region_size - n
        df2 = 2 * region_size - n - 1

        #Get the F-value
        F = FTest(chiSq_m, chiSq_n, df2)

        #Get the p-value
        p = 2 * betainc(0.5 * df2, 0.5 * df1, df2 / (df2 + df1 * F))

        if p > p_threshold:
            lambdaAbsorption = extendedLambda[left: right + 1]
            lambdaAbsorption_norm = (2 * lambdaAbsorption - Lambda_L - Lambda_U) / (Lambda_U - Lambda_L)
            return np.array([legendreLinCom(fit_n, x) for x in lambdaAbsorption_norm])

    raise ValueError('Unable to establish the continuum flux using Legendre polynomials, try increasing the max_order parameter.')

def FTest(chiSq1: float, chiSq2: float, df: int) -> float:
    """
    Calculates the F-test result for two chi-square values and degrees of freedom.
    
    Args:
        chiSq1 (float): The first chi-square value.
        chiSq2 (float): The second chi-square value.
        df (int): The degrees of freedom.

    Returns:
        float: The F-test result.
    """

    return (chiSq1 - chiSq2) / (chiSq2 / df)

def legendreChiSq(coeff: np.ndarray, x: np.ndarray, y: np.ndarray, sigy: np.ndarray) -> float:
    """
    Calculates the chi-squared error between two fits.
        
    This function works by looping through each element of the x-values (derived from the wavelength) in the 
    window and performing the following steps for each element:    
        - Finding the continuum, which is a linear combination of Legendre polynomials up to a certain degree M. This 
            is done in an inner loop.
        - Using the continuum and the y-values and uncertainties (I, sig I) to calculate the chi-squared error.
        - Repeating these steps for the next x-value.

    Args:
        coeff (np.ndarray): The coefficients of the Legendre polynomial fit.
        x (np.ndarray): The x-values of the data (the wavelength in this context).
        y (np.ndarray): The y-values of the data (the flux in this context).
        sigy (np.ndarray): The uncertainties in the y-values.

    Returns:
        The chi-squared error.
    """

    if len(y) != len(x) or len(sigy) != len(x):
        raise ValueError('y and sigy must have the same length as x.')

    chiSq = 0
    for i in range(len(x)):
        legSum = legendreLinCom(coeff, x[i])
        chiSq += (y[i] - legSum)**2 / (sigy[i])**2
    
    return chiSq

def legendreLinCom(coeff: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Calculates a linear combination of Legendre Polynomials up to a given degree.
    
    Parameters:
        coeff (np.ndarray): Coefficients of the Legendre Polynomials.
        x (np.ndarray): Array of x values at which to evaluate the polynomial.

    Returns:
        Values of the polynomial at the given x values.
    """

    degree = np.arange(len(coeff))
    legendreValues = np.zeros(len(coeff))

    for j in range(len(degree)):
        legendreValues[j] = eval_legendre(degree[j], x)

    return np.dot(coeff, legendreValues)

def fluxDec(i: int, flux: np.ndarray, yC: np.ndarray, sigFlux: np.ndarray, sig_yC: np.ndarray) -> Tuple[float, float]:
    """
    Calculate the flux decrement at a pixel i.

    If the flux decrement satisfies some detection threshold at a pixel, 
    go left and right from the pixel to find the points where the continuum 
    recovers sufficiently. 

    Parameters:
        i (int): the index of the pixel to calculate the flux decrement for.
        flux (np.ndarray): Array of flux values.
        yC (np.ndarray): Array of continuum values.
        sigFlux (np.ndarray): Array of uncertainties in the flux values.
        sig_yC (np.ndarray): Array of uncertainties in the continuum values.
        
    Returns:
        Two values, the flux decrement at the given pixel and the corresponding uncertainty.
    """
    
    # Check that the input arrays have the same length
    if not (len(flux) == len(yC) == len(sigFlux) == len(sig_yC)):
        raise ValueError("Input arrays must have the same length")
    
    # Flux decrement per pixel
    # -ve for emission since flux > continuum. +ve for absorption.
    with np.errstate(divide='ignore', invalid='ignore'): #So Numpy ignores division by zero errors, will return NaN
        D = 1 - (flux[i] / yC[i])
        D = np.nan_to_num(D)
    
    # Uncertainty in the flux decrement
    deltaD = (flux[i] / yC[i]) * np.sqrt((sigFlux[i] / flux[i])**2 + (sig_yC[i] / yC[i])**2)

    return D, deltaD



def getP(i: int, Lambda: np.ndarray, R: np.ndarray, resolution_element: int = 3) -> np.ndarray:
    """
    Calculates the instrumental spread function around a pixel i using a discrete or continuous Gaussian model.
    
    Note:
        The continuous method is more accurate than the discrete method, since it models the ISF as 
        a continuous function rather than a discrete sum. However, it is also more computationally expensive, 
        since it requires the evaluation of a Gaussian function at every pixel within a certain range. The discrete 
        method is less accurate, but much faster to compute.
    
    Args:
        i (int): Index of the pixel.
        Lambda (np.ndarray): Array of wavelength values.
        R (np.ndarray): Array of resolving powers.
        resolution_element (int): The size of the resolution element in pixels. Defaults to 3.
    
    Returns:
        Array of normalized instrumental spread function values.
    """

    #Define J = 2*resolution_element
    J0 = 2 * resolution_element 
    #Expression for the uncertainty in the ISF
    sigISF = Lambda[i] / (2.35 * R[i])

    #Range of pixels over which the ISF is computed (i - J0 to i + J0)
    x = np.zeros(2*J0+1)
    #Gaussian model of the ISF
    P = np.zeros(2*J0+1)

    #Compute the x values and corresponding P_n (n is j here) around the pixel i
    for j in range(2*J0+1):
        #If the resolution element goes out of bounds of spectrum, end the loop
        if i + j - J0 >= len(Lambda):
            break
        #If not, find the value for the ISF
        x[j] = (Lambda[i] - Lambda[i + j - J0]) / sigISF ### Not j - 1 since j starts at 0, not 1
        P[j] = np.exp(-x[j] ** 2)

    #Return the normalized instrumental spread function
    return P / np.sum(P)


def optimizedResEleEW(i: int, Lambda: np.ndarray, flux: np.ndarray, yC: np.ndarray, sigFlux: np.ndarray, sig_yC: np.ndarray,
    R: np.ndarray, resolution_element: int = 3) -> Tuple[int, int]:
    """
    Calculates the equivalent width per resolution element using the optimized method.

    Args:
        i (int): Index of the pixel.
        Lambda (np.ndarray): Array of wavelength values.
        flux (np.ndarray): Array of flux values.
        yC (np.ndarray): Array of continuum values.
        sigFlux (np.ndarray): Array of flux uncertainties.
        sig_yC (np.ndarray): Array of continuum uncertainties.
        R (np.ndarray): Array of resolving powers.
        resolution_element (int): The size of the resolution element in pixels. Defaults to 3.

    Returns:
        Two values, the equivalent width per resolution element and its uncertainty.
    """

    J0 = 2*resolution_element

    #Determine the pixel width
    deltaLambda = Lambda[i+1] - Lambda[i]

    #Get the array containing P from i - J0 to i + J0
    P = getP(i, Lambda, R, resolution_element=resolution_element)

    #Compute P^2
    sqP = np.sum(P**2)

    #Calculating EW per res element
    eqWidth = 0
    for j in range(2*J0+1):
        eqWidth += P[j] * fluxDec(i + j - J0, flux, yC, sigFlux, sig_yC)[0]

    #EW per res element. Multiplying by the constant coefficient
    coeff = (deltaLambda / sqP)
    eqWidth = coeff * eqWidth

    #Uncertainty - EW per res element
    deltaEqWidth = 0
    for j in range(2*J0+1):
        deltaEqWidth += P[j]**2 * fluxDec(i + j - J0, flux, yC, sigFlux, sig_yC)[1]**2
    
    #Uncertainty - EW per res element
    deltaEqWidth = coeff * np.sqrt(deltaEqWidth)

    return eqWidth, deltaEqWidth

def optimizedFeatureLimits(i: int, Lambda: np.array, flux: np.array, yC: np.array, sigFlux: np.array, sig_yC: np.array, R: np.ndarray, 
    flux_threshold: float = 0.5, resolution_element: int = 3) -> Tuple[int, int]:
    """
    Finds the left and right limits of a feature based on the recovery of the flux.
    
    Parameters:
        i (int): Index of the pixel.
        Lambda (np.array): Array of wavelength values.
        flux (np.array): Array of flux values.
        yC (np.array): Array of continuum values.
        sigFlux (np.array): Array of flux uncertainties.
        sig_yC (np.array): Array of continuum uncertainties.
        R (np.ndarray): Array of resolving powers.
        flux_threshold (float): Threshold of flux recovery for determining feature limits.
        resolution_element (int): The size of the resolution element in pixels. Defaults to 3.
    
    Returns:
        left_index (int): Index of the left limit of the feature.
        right_index (int): Index of the right limit of the feature.
    """
    

    # Define variables for left and right indices
    left_index, right_index = i, i 
    
    #Scan blueward to find the left limit of the feature
    while left_index >= 0:
        eqWidth, deltaEqWidth = optimizedResEleEW(left_index, Lambda, flux, yC, sigFlux, sig_yC, R, resolution_element)

        # Does the flux recover sufficiently at this pixel?
        if abs(eqWidth / deltaEqWidth) <= flux_threshold:
            #If so, exit the loop
            left_index -= 1
            break
        left_index -= 1 #If not, start again for the preceding pixel i.e. decrement the pixel by 1
        
    #Scan redward to find the right limit of the feature
    while right_index < len(Lambda) - 1:  #Max right_index allowed is less than the ISF width
        eqWidth, deltaEqWidth = optimizedResEleEW(right_index, Lambda, flux, yC, sigFlux, sig_yC, R, resolution_element)
        
        #Does the flux recover sufficiently at this pixel?
        if abs(eqWidth / deltaEqWidth) <= flux_threshold:
            #If so, exit the loop
            right_index += 1
            break  
        right_index += 1 # If not, start again for the next pixel i.e. increment the pixel by 1
    
    #Handle cases where the function cannot find valid feature limits
    if left_index < 0 or right_index >= len(Lambda) - 1:
        raise ValueError("Could not find valid feature limits.")
    
    #Return the INDICES (NOT WAVELENGTH!!) that form the limits of the feature
    return left_index, right_index

def MgII(Lambda: np.array, flux: np.array, yC: np.array, sigFlux: np.array, sig_yC: np.array, R: np.ndarray, 
    flux_threshold: float = 0.5, resolution_element: int = 3) -> Tuple[int, int]:
    """ 
    Calculates the equivalent width and associated uncertainties of Mg-II doublet absorption features in a given spectrum.

    Parameters:
        Lambda (np.ndarray): Wavelength array of the spectrum
        flux (np.ndarray): Flux array of the spectrum
        yC (np.ndarray): Continuum flux array of the spectrum
        sigFlux (np.ndarray): Array of flux uncertainties
        sig_yC (np.ndarray): Array of continuum flux uncertainties
        R (np.ndarray): Resolution array of the spectrum
        flux_threshold (float): Threshold of flux recovery for determining feature limits.
        resolution_element (int): The size of the resolution element in pixels. Defaults to 3.
     
    Returns:
        Mg2796 (np.ndarray): Array of lower limit wavelength values for the Mg-II 
        Mg2803 (np.ndarray): Array of upper limit wavelength values for each Mg-II feature detected
        EW2796 (np.ndarray): Array of equivalent widths for each Mg-II 2796 feature detected
        EW2803 (np.ndarray): Array of equivalent widths for each Mg-II 2803 feature detected
        deltaEW2796 (np.ndarray): Array of uncertainties in equivalent widths for each Mg-II 2796 feature detected
        deltaEW2803 (np.ndarray): Array of uncertainties in equivalent widths for each Mg-II 2803 feature detected
    """

    #Define an empty array to hold the line limits, EW and associated uncertainty
    Mg2796, Mg2803, EW2796, EW2803, deltaEW2796, deltaEW2803 = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    
    #Instrumental spread function half-width
    J0 = 2 * resolution_element

    #Pad the arrays with zeros to avoid the risk of the code running out of bounds in the spectrum
    extendedLambda = np.pad(Lambda, (J0, J0), 'edge')
    extendedFlux = np.pad(flux, (J0, J0), 'constant', constant_values=yC[0])
    extendedyC = np.pad(yC, (J0, J0), 'edge')
    extendedsigFlux = np.pad(sigFlux, (J0, J0), 'constant', constant_values=sigFlux[0])
    extendedsig_yC = np.pad(sig_yC, (J0, J0), 'edge')
    extendedR = np.pad(R, (J0, J0), 'edge')

    #Run through every pixel in the spectrum
    i = J0  #Skip over the extra pixels added so the code does not run out of bounds in the spectrum 
    while i < len(extendedLambda) - J0:
        flag = 0
        #Find the equivalent width at the pixel using the Optimized Method
        eqWidth1, deltaEqWidth1 = optimizedResEleEW(i, extendedLambda, extendedFlux, extendedyC, extendedsigFlux, extendedsig_yC, extendedR, resolution_element)

        #Check if the pixel satisfies the selection criterion
        #But first change them to fall fellow the threshold if they are not finite, so as to avoid warnings.
        eqWidth1 = 1e-3 if (eqWidth1 > 0)==False else eqWidth1
        deltaEqWidth1 = 1e3 if (deltaEqWidth1 > 0)==False else deltaEqWidth1

        if eqWidth1/deltaEqWidth1 > 5:
            #Congrats! We have located an absorption feature. We need to ensure the absorption feature is indeed Mg-II. 
            #If we assume this feature to be the 2796 line, there must be a second absorption feature at the equilibrium separation.
            #To look for such a pixel, we first find the redshift and then the equilibrium separation.
            z = extendedLambda[i] / 2796.35 - 1

            #The separation is then z * 7.18 (which is the separation of the troughs at zero redshift) 
            sep = (z + 1) * 7.18

            #Find the index of the first element from the wavelength list that is greater than the required separation. 
            try:
                index = next(j for j, val in enumerate(extendedLambda) if val > extendedLambda[i] + sep)
                if index + J0 + 10 >= len(extendedLambda):   #Ensure we do not run out of bounds
                    i += 1
                    continue
                
            except:
                i += 1
                continue

            #Find the equivalent width and the corresponding uncertainty at a range around the second pixel
            for k in range(index-2, index+3):
                
                #Find the pixel eq width around the second pixel and check if there is a second absorption system
                eqWidth2, deltaEqWidth2 = optimizedResEleEW(k, extendedLambda, extendedFlux, extendedyC, extendedsigFlux, extendedsig_yC, extendedR, resolution_element)

                if eqWidth2/deltaEqWidth2 > 3:

                    #Get the wavelength range of each absorption range now that both the systems are stat. sig.
                    line1B, line1R = optimizedFeatureLimits(i, extendedLambda, extendedFlux, extendedyC, extendedsigFlux, extendedsig_yC, extendedR, flux_threshold, resolution_element)
                    line2B, line2R = optimizedFeatureLimits(k, extendedLambda, extendedFlux, extendedyC, extendedsigFlux, extendedsig_yC, extendedR, flux_threshold, resolution_element)

                    #Calculate the total EW over the two features
                    EW1, sigEW1 = apertureEW(line1B, line1R, extendedLambda, extendedFlux, extendedyC, extendedsigFlux, extendedsig_yC)
                    EW2, sigEW2 = apertureEW(line2B, line2R, extendedLambda, extendedFlux, extendedyC, extendedsigFlux, extendedsig_yC)

                    #Convert EW to rest frame
                    EW1, sigEW1 = EW1/(1+z), sigEW1/(1+z)
                    z2 = 0.5 * (extendedLambda[line2B] + extendedLambda[line2R])/2796
                    EW2, sigEW2 = EW2/z2, sigEW2/z2

                    if EW1/sigEW1 > 5 and EW2/sigEW2 > 3 and EW1/EW2 < 2.1 and EW1/EW2 > 0.95:
                        Mg2796 = np.append(Mg2796, [line1B - J0, line1R - J0])
                        Mg2803 = np.append(Mg2803, [line2B - J0, line2R - J0])
                        EW2796 = np.append(EW2796, [EW1])
                        EW2803 = np.append(EW2803, [EW2])
                        deltaEW2796 = np.append(deltaEW2796, [sigEW1])
                        deltaEW2803 = np.append(deltaEW2803, [sigEW2])

                        i = line2R + 1
                        flag = 1
                        break

            if flag == 1:
                continue

        i += 1

    return Mg2796, Mg2803, EW2796, EW2803, deltaEW2796, deltaEW2803


