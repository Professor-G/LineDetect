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
from lmfit import Model

class Continuum:
    """
    Generates the continuum and continuun error.
   
    Args:
        Lambda (np.ndarray): Array of wavelengths.
        flux (np.ndarray): Array of self.flux values.
        flux_err (np.ndarray): Array of uncertainties in the self.flux values.          
        halfWindow (int): The half-window size to use for the smoothing procedure.
            If this is a list/array of integers, then the continuum will be calculated
            as the median curve across the fits across all half-window sizes in the list/array.
            Defaults to 25.
        N_sig (int): Defaults to 3.
        region_size (int): The size of the region to apply the polynomial fitting. Defaults to 150 pixels.        resolution_element (int): Defaults to 3.

    Methods:
        estimate:
        median_filter:
        legendreContinuum:

    """

    def __init__(self, Lambda, flux, flux_err, halfWindow=25, N_sig=3, region_size=150, resolution_element=3):
        self.Lambda = Lambda
        self.flux = flux
        self.flux_err = flux_err
        self.halfWindow = halfWindow
        self.N_sig = N_sig
        self.region_size = region_size
        self.resolution_element = resolution_element

        self.size = len(self.Lambda)

        mask = np.where(np.isfinite(self.flux))[0]
        if len(mask) == 0:
            raise ValueError('WARNING: No non-finite values detected in the flux array!')
        if len(mask) != len(self.flux):
            print(f"WARNING: {len(mask)} non-finite values detected in the flux array, masking away...")
            self.Lambda, self.flux, self.flux_err = self.Lambda[mask], self.flux[mask], self.flux_err[mask]
        
    def estimate(self, fit_legendre=True):
        """
        Fits the spectra continuum using a robust moving median followed by a Legendre polynomial fit.

        Args:
            fit_legendre (bool): Whether to fit with Legendre polynomials for a more robust estimate. Defaults to True.
        
        Returns:
            Generates the continuum and continuum_err class attributes.
        """
        
        #Apply the robust median filter first, legendre continuum fitter second
        self.median_filter(); self.legendreContinuum(max_order=20, p_threshold=0.05) if fit_legendre else None
        
    def median_filter(self):
        """
        Smooths out the flux array with a robust median-based filter

        Returns:
            Creates the continuum and continuum_err attributes.
        """

        if isinstance(self.halfWindow, int):
            yC, sig_yC = np.empty(self.size), np.empty(self.size)
            # Loop to scan through each pixel in the spectrum
            for i in range(self.size):
                # Condition for the blue region of the spectrum where there is insufficient data to the left of the pixel
                if i < self.halfWindow:
                    yC[i] = np.median(self.flux[0: i + self.halfWindow + 1])
                    sig_yC[i] = 0.05 #This is the error of the medium continuum, can also be the median(flux_err)
                    continue

                # Condition for the red region of the spectrum where there is insufficient data to the right of the pixel
                if i >= self.size - self.halfWindow:
                    yC[i] = np.median(self.flux[i - self.halfWindow : ])
                    sig_yC[i] = 0.05
                    continue

                # Not the end cases
                yC[i] = np.median(self.flux[i - self.halfWindow : i + self.halfWindow + 1])
                sig_yC[i] = 0.05
            
            self.continuum, self.continuum_err = yC, sig_yC

        else:
            yC_list, sig_yC_list = [], []
            for window_size in self.halfWindow:
                yC, sig_yC = np.empty(self.size), np.empty(self.size)
                # Loop to scan through each pixel in the spectrum
                for i in range(0, self.size):
                    # Condition for the blue region of the spectrum where there is insufficient data to the left of the pixel
                    if i < window_size:
                        yC[i] = np.median(self.flux[0: i + self.halfWindow + 1])
                        sig_yC[i] = 0.05
                        continue

                    # Condition for the red region of the spectrum where there is insufficient data to the right of the pixel
                    if i >= self.size - window_size:
                        yC[i] = np.median(self.flux[i - self.halfWindow : ])
                        sig_yC[i] = 0.05
                        continue

                    # Not the end cases
                    yC[i] = np.median(self.flux[i - window_size : i + window_size + 1])
                    sig_yC[i] = 0.05
                
                yC_list.append(yC); sig_yC_list.append(sig_yC)
                
            #Average across individual pixels
            yC, sig_yC = np.median(yC_list, axis=0), np.median(sig_yC_list, axis=0)

        self.continuum, self.continuum_err = yC, sig_yC

        return
    
    def legendreContinuum(self, max_order, p_threshold):
        """
        Fits a Legendre polynomial to a spectrum to locate absorption and/or emission features.

        The function identifies absorption or emission features in the input spectrum using the `featureFinder` function,
        then extends thefl spectrum on both sides to avoid running out of bounds. For each feature, it selects a region
        of size region_size pixels around it and fits a Legendre polynomial to this region. If there are two features less than region_size pixels
        apart, the function computes the gap between them and includes the pixels between the features in the fitting array.
        The function returns an array of the fitted continuum values.

        Args:
            max_order (int): The maximum order of the Legendre polynomial to fit. Defaults to 20.
            p_threshold (float): The p-value threshold for the F-test. If the p-value is greater than this threshold,
                the fit is considered acceptable and the continuum level is returned. Defaults to 0.05.

        Returns:
            Updates the existing continuum and continuum_err class attributes.
        """

        # Find the regions of absorption
        featureRange = featureFinder(self.Lambda, self.flux, self.continuum, self.flux_err, self.continuum_err, N_sig=self.N_sig)
        
        # If there are no absorption lines, return the unchanged continuum array
        if featureRange.size == 0:
            return 

        clean_pixels = []

        #Iterate through the list of absorption features and start fitting 
        #Define an array to hold the regions within 
        for i in range(len(featureRange) // 2):
            clean_pixels.clear()

            # Start and end of the current absorption feature
            start = featureRange[2 * i]
            end = featureRange[2 * i + 1]

            # We now have an absorption feature selected. 
            # The next step is to choose 150 clean pixels (i.e. pixels devoid of any absorption or emission) on either side of the abs line.
            # Sometimes an absorption feature might have another absorption feature within 150 pixels of each other.
            # If so we skip the second feature and keep going until we hit 150 pixels.
            # We need to get 150 pixels on the left and right side of each absorption feature.
            # Thus this whole shebang needs to be done twice.

            ##########   Left   ############
            # Let us assume the left boundary is 150 pixels to the left of the absorption line
            left_boundary = start - 150
            # countLeft keeps count of how many pixels we have covered from the left end of the absorption line.
            # This helps tremendously when we skip over a second absorption line and have to keep track of how many more pixels we need.
            countLeft = 150
            
            # Start going to the left until we hit 150
            j = i - 1
            while True:

                # If we have reached the first absorption line, 
                # check if the we have enough pixels to the left of it.
                # If we don't, stop at the very first pixel.
                if j < 0:
                    left_boundary = max(left_boundary, 0)
                    clean_pixels.extend(range(left_boundary, start + 1))
                    break

                # If we have enough pixels to the left of the current absorption, break, and go to the next absorption line 
                elif start - featureRange[2 * j + 1] > countLeft and start - featureRange[2 * j + 1] > 2*self.resolution_element and left_boundary - featureRange[2 * j + 1] > 2*self.resolution_element:
                    clean_pixels.extend(range(left_boundary - 2*self.resolution_element, start))
                    break
                
                elif start - featureRange[2 * j + 1] > countLeft:
                    clean_pixels.extend(range(featureRange[2 * j + 1], start + 1))
                    countLeft -= start - featureRange[2 * j + 1] + 2*self.resolution_element

                # If we don't have enough pixels and we are not at the first absorption line,
                # cut down the number of pixels we have left.  
                else:
                    clean_pixels.extend(range(featureRange[2 * j + 1], start + 1))
                    countLeft -= start - featureRange[2 * j + 1]

                # For the next iteration, the start pixel becomes the left end of the previous absorption line
                start = featureRange[2 * j]
                # Update j and the left boundary. 
                # This is where countLeft becomes critical.
                left_boundary = start - countLeft
                j -= 1

            ##########   Right   ############
            # Same stuff as the left
            right_boundary = end + 150
            countRight = 150

            j = i + 1
            leftEdge = 0
            while True:
                if j > len(featureRange) // 2 - 1:
                    right_boundary = min(right_boundary, len(Lambda) - 1)
                    clean_pixels.extend(range(end, right_boundary + 1))
                    break

                elif featureRange[2 * j] - end > countRight and featureRange[2 * j] - end > 2*self.resolution_element:
                    clean_pixels.extend(range(end, right_boundary + 1 + 2*self.resolution_element))
                    break

                elif featureRange[2 * j] - end > countRight:
                    clean_pixels.extend(range(end, featureRange[2 * j] + 1))
                    countRight -= featureRange[2 * j] - end + 2*self.resolution_element

                else:
                    clean_pixels.extend(range(end, featureRange[2 * j] + 1))
                    countRight -= featureRange[2 * j] - end

                end = featureRange[2 * j + 1]
                right_boundary = end + countRight
                j += 1

            #Find the functional fit of the continuum in this range
            result = legendreFit(clean_pixels, self.Lambda, self.flux, self.sigFlux, region_size=self.region_size, max_order=max_order, p_threshold=p_threshold)

            if result is not None:
                if clean_pixels[0] >= leftEdge:
                    self.continuum[clean_pixels[0] : clean_pixels[-1] + 1] = result[0]
                    self.continuum_err[clean_pixels[0] : clean_pixels[-1] + 1] = result[1]
                    leftEdge = clean_pixels[-1]
                else:
                    self.continuum[leftEdge : clean_pixels[-1] + 1] = result[0]
                    self.continuum_err[clean_pixels[0] : clean_pixels[-1] + 1] = result[1]

        return 

def legendreFit(indices, Lambda, flux, sigFlux, region_size, max_order, p_threshold):
    """
    Fits a Legendre polynomial to a given spectrum so as to estimate the continuum.

    Args:
        indices (np.ndarray): The x-values of the spectrum to be fit.
        Lambda (np.ndarray): The x-values of the extended spectrum.
        flux (np.ndarray): The y-values of the spectrum to be fit.
        sigFlux (np.ndarray): The uncertainties in the y-values.
        region_size (int): The number of points on either side of a given point to use in the fit.
        max_order (int): The maximum order of the Legendre polynomial to fit. Defaults to 20.
        p_threshold (float): The p-value threshold for the F-test. If the p-value is greater than this threshold,
            the fit is considered acceptable and the continuum level is returned. Defaults to 0.05.

    Returns:
        The continuum level of the spectrum.
    """

    fitLambda, fitFlux, fitsigFlux = np.array(Lambda[indices]), np.array(flux[indices]), np.array(sigFlux[indices])

    #Convert the x-axis to the domain [-1, 1]
    Lambda_L, Lambda_U = fitLambda[0], fitLambda[-1]
    fitLambda = (2*fitLambda - Lambda_L - Lambda_U) / (Lambda_U - Lambda_L)

    ##Start at the first order Legendre Polynomial and compare it with the 0th order, and so on.
    for n in range(1, max_order + 1):
        #Fit the window with Legendre polynomial of degree n - 1 = m 
        fit_m = np.polynomial.legendre.legfit(fitLambda, fitFlux, n - 1)

        #Fit the window with Legendre polynomial of degree n
        fit_n = np.polynomial.legendre.legfit(fitLambda, fitFlux, n)
            
        # Construct the Vandermonde matrix
        vander = np.polynomial.legendre.legvander(fitLambda, n)

        # Compute the covariance matrix
        covariance = np.linalg.inv(vander.T @ vander)

        #Find chi square for both of the fits
        chiSq_m = legendreChiSq(fit_m, fitLambda, fitFlux, fitsigFlux)
        chiSq_n = legendreChiSq(fit_n, fitLambda, fitFlux, fitsigFlux)

        df1 = 2 * region_size - n
        df2 = 2 * region_size - n - 1

        #Get the F-value
        F = FTest(chiSq_m, chiSq_n, df2)

        #Get the p-value
        p = 2 * betainc(0.5*df2, 0.5*df1, df2/(df2 + df1 * F))

        if p > p_threshold:
            left, right = indices[0], indices[-1]

            #Define the region over which the functional fit has to be applied
            lambdaAbsorption = np.array(Lambda[left : right + 1])

            #Convert from lambda to x
            lambdaAbsorption = (2*lambdaAbsorption - Lambda_L - Lambda_U) / (Lambda_U - Lambda_L)
            
            #Find the continuum in this wavelength range
            absFit, absFitErr = np.zeros(len(lambdaAbsorption)), np.zeros(len(lambdaAbsorption))

            for i in range(len(lambdaAbsorption)):
                absFit[i] = legendreLinCom(fit_n, lambdaAbsorption[i])
                for j in range(len(fit_n)):
                    for k in range(len(fit_n)):
                        absFitErr[i] += covariance[j][k] ** 2 * eval_legendre(j, lambdaAbsorption[j]) * eval_legendre(k, lambdaAbsorption[k])
                absFitErr[i] = np.sqrt(absFitErr[i])
            
            return [absFit, absFitErr]

    raise ValueError('Unable to establish the continuum flux using Legendre polynomials, try increasing the max_order parameter and/or decreasing the p_threshold.')

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
    N_sig: float = 0.5, resolution_element: int = 3) -> Tuple[int, int]:
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
        N_sig (float): Threshold of flux recovery for determining feature limits.
        resolution_element (int): The size of the resolution element in pixels. Defaults to 3.
    
    Returns:
        left_index (int): Index of the left limit of the feature.
        right_index (int): Index of the right limit of the feature.
    """

    # Define variables for left and right indices
    left_index = right_index = i 
    
    #Scan blueward to find the left limit of the feature
    while left_index >= 0:
        eqWidth, deltaEqWidth = optimizedResEleEW(left_index, Lambda, flux, yC, sigFlux, sig_yC, R, resolution_element)

        # Does the flux recover sufficiently at this pixel?
        if abs(eqWidth / deltaEqWidth) <= N_sig:
            #If so, exit the loop
            left_index -= 1
            break
        left_index -= 1 #If not, start again for the preceding pixel i.e. decrement the pixel by 1
        
    #Scan redward to find the right limit of the feature
    while right_index < len(Lambda) - 1:  #Max right_index allowed is less than the ISF width
        eqWidth, deltaEqWidth = optimizedResEleEW(right_index, Lambda, flux, yC, sigFlux, sig_yC, R, resolution_element)
        
        #Does the flux recover sufficiently at this pixel?
        if abs(eqWidth / deltaEqWidth) <= N_sig:
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
    N_sig_1: float = 5, N_sig_2: float = 3, resolution_element: int = 3, rest_wavelength_1: float = 2796.35,
    rest_wavelength_2: float = 2803.53) -> Tuple[int, int]:
    """ 
    Calculates the equivalent width and associated uncertainties of Mg-II doublet absorption features in a given spectrum.

    Parameters:
        Lambda (np.ndarray): Wavelength array of the spectrum
        flux (np.ndarray): Flux array of the spectrum
        yC (np.ndarray): Continuum flux array of the spectrum
        sigFlux (np.ndarray): Array of flux uncertainties
        sig_yC (np.ndarray): Array of continuum flux uncertainties
        R (np.ndarray): Resolution array of the spectrum
        N_sig_1 (float): Threshold of flux recovery for determining feature limits.
        N_sig_2 (float): Defaults to 3.
        resolution_element (int): The size of the resolution element in pixels. Defaults to 3.
        rest_wavelength_1 (float):
        rest_wavelength_2 (float):
     
    Returns:
        Mg2796 (np.ndarray): Array of lower limit wavelength values for the Mg-II 
        Mg2803 (np.ndarray): Array of upper limit wavelength values for each Mg-II feature detected
        EW2796 (np.ndarray): Array of equivalent widths for each Mg-II 2796 feature detected
        EW2803 (np.ndarray): Array of equivalent widths for each Mg-II 2803 feature detected
        deltaEW2796 (np.ndarray): Array of uncertainties in equivalent widths for each Mg-II 2796 feature detected
        deltaEW2803 (np.ndarray): Array of uncertainties in equivalent widths for each Mg-II 2803 feature detected
    """
    doublet_separation = abs(rest_wavelength_2 - rest_wavelength_1) #7.18
    #Define an empty array to hold the line limits, EW and associated uncertainty
    Mg2796, Mg2803, EW2796, EW2803, deltaEW2796, deltaEW2803 = [],[],[],[],[],[]
    
    #Instrumental spread function half-width
    J0 = 2 * resolution_element

    #Run through every pixel in the spectrum
    i = 0
    while i < len(Lambda):
        flag = 0
        #Find the equivalent width at the pixel using the Optimized Method
        eqWidth1, deltaEqWidth1 = optimizedResEleEW(i, Lambda, flux, yC, sigFlux, sig_yC, R, resolution_element)

        #Check if the pixel satisfies the selection criterion
        #But first change them to fall fellow the threshold if they are not finite, so as to avoid warnings.
        eqWidth1 = 1e-3 if (eqWidth1 > 0)==False else eqWidth1
        deltaEqWidth1 = 1e3 if (deltaEqWidth1 > 0)==False else deltaEqWidth1

        if eqWidth1 / deltaEqWidth1 > N_sig_1:
            #Congrats! We have located an absorption feature. We need to ensure the absorption feature is indeed Mg-II. 
            #If we assume this feature to be the 2796 line, there must be a second absorption feature at the equilibrium separation.
            #To look for such a pixel, we first find the redshift and then the equilibrium separation.
            z = Lambda[i] / rest_wavelength_1 - 1

            #The separation is then z * 7.18 (which is the separation of the troughs at zero redshift) 
            sep = (z + 1) * doublet_separation

            #Find the index of the first element from the wavelength list that is greater than the required separation. 
            try:
                index = next(j for j, val in enumerate(Lambda) if val > Lambda[i] + sep)
                i += 1
                
            except:
                i += 1
                continue

            #Find the equivalent width and the corresponding uncertainty at a range around the second pixel
            for k in range(index-2, index+3):
                
                #Find the pixel eq width around the second pixel and check if there is a second absorption system
                eqWidth2, deltaEqWidth2 = optimizedResEleEW(k, Lambda, flux, yC, sigFlux, sig_yC, R, resolution_element)

                if eqWidth2 / deltaEqWidth2 > N_sig_2:

                    #Get the wavelength range of each absorption range now that both the systems are stat. sig.
                    line1B, line1R = optimizedFeatureLimits(i, Lambda, flux, yC, sigFlux, sig_yC, R, N_sig_2, resolution_element)
                    line2B, line2R = optimizedFeatureLimits(k, Lambda, flux, yC, sigFlux, sig_yC, R, N_sig_2, resolution_element)

                    if line1B == line2B:
                        EW1, sigEW1, EW2, sigEW2 = lineFit(line1B, line2R, Lambda, flux, rest_wavelength_1, rest_wavelength_2) # This function already converst the EW to restframe!
                    else:
                        #Calculate the total EW over the two features
                        EW1, sigEW1 = apertureEW(line1B, line1R, Lambda, flux, yC, sigFlux, sig_yC)
                        EW2, sigEW2 = apertureEW(line2B, line2R, Lambda, flux, yC, sigFlux, sig_yC)

                        #Convert EW to rest frame
                        EW1, sigEW1 = EW1 / (1 + z), sigEW1 / (1 + z)
                        z2 = 0.5 * (Lambda[line2B] + Lambda[line2R]) / rest_wavelength_2 # z2 is equal to 1 + z2
                        EW2, sigEW2 = EW2 / z2, sigEW2 / z2

                    if (EW1 / sigEW1 > N_sig_1) and (EW2 / sigEW2 > N_sig_2) and (0.95 < EW1 / EW2 < 2.1):
                        Mg2796.extend([line1B, line1R]); Mg2803.extend([line2B, line2R])
                        EW2796.append(EW1); EW2803.append(EW2)
                        deltaEW2796.append(sigEW1); deltaEW2803.append(sigEW2)

                        i = line2R + 1
                        flag = 1
                        break

            if flag == 1:
                continue

        i += 1

    return Mg2796, Mg2803, EW2796, EW2803, deltaEW2796, deltaEW2803


def doubleGaussian(x, amplitude_1, amplitude_2, sigma_1, sigma_2, x0_1, x0_2):
    """
    Calculates the value of a double Gaussian function at a given x coordinate.
    
    Parameters:
        x (float): The x coordinate.
        amplitude_1 (float): Amplitude of the first Gaussian component.
        amplitude_2 (float): Amplitude of the second Gaussian component.
        sigma_1 (float): Standard deviation of the first Gaussian component.
        sigma_2 (float): Standard deviation of the second Gaussian component.
        x0_1 (float): Mean of the first Gaussian component.
        x0_2 (float): Mean of the second Gaussian component.
        
    Returns:
        float: The value of the double Gaussian function at the given x coordinate.
    """

    return np.exp((-0.5*(x-x0_1)**2)/sigma_1**2)*amplitude_1 + np.exp((-0.5*(x-x0_2)**2)/sigma_2**2)*amplitude_2 + 1

def lineFit(index1, index2, Lambda, flux, rest_wavelength_1, rest_wavelength_2):
    """
    Fits a line using a double Gaussian model and returns the equivalent width and errors.

    Parameters:
        index1 (int): Start index of the line region.
        index2 (int): End index of the line region.
        Lambda (array-like): Array of wavelengths.
        flux (array-like): Array of flux values.
        rest_wavelength_1 (float):
        rest_wavelength_2 (float):

    Returns:
        tuple: A tuple containing the equivalent widths and errors of the two Gaussian components:
            - EW1 (float): Equivalent width of the first Gaussian component.
            - deltaEW1 (float): Error in the equivalent width of the first Gaussian component.
            - EW2 (float): Equivalent width of the second Gaussian component.
            - deltaEW2 (float): Error in the equivalent width of the second Gaussian component.
    """

    step = (index2-index1)/4
    x0_1 = Lambda[round(step)+index1]
    x0_2 = Lambda[3*round(step)+index1]
    z = x0_1/rest_wavelength_1 - 1
    wavelength = Lambda[index1 - 30:index2 + 30]
    fitFlux = np.append(np.ones(30), flux[index1:index2])
    fitFlux = np.append(fitFlux, np.ones(30))

    fmodel = Model(doubleGaussian)
    params = fmodel.make_params(amplitude_1=-0.5, amplitude_2=-0.5, sigma_1=2, sigma_2=2, x0_1=x0_1, x0_2=x0_2)

    result = fmodel.fit(fitFlux, params, x=wavelength)
    coeff = result.best_values
    perr = np.sqrt(np.diag(result.covar))

    # Condition to check if the line is unresolved
    if coeff['sigma_1'] + perr[2] < 1.46:
        params['sigma_1'].vary = False
        result = fmodel.fit(fitFlux, params, x=wavelength)

    if coeff['sigma_2'] + perr[3] < 1.46:
        params['sigma_2'].vary = False
        result = fmodel.fit(fitFlux, params, x=wavelength)

    EW1 = -np.sqrt(2 * np.pi) * coeff['sigma_1'] * coeff['amplitude_1']
    deltaEW1 = np.sqrt((perr[2] / coeff['sigma_1'])**2 + (perr[0] / coeff['amplitude_1'])**2 - (2 * perr[0] * perr[2])**2)
    z1 = coeff['x0_1'] / rest_wavelength_1 - 1

    EW2 = -np.sqrt(2 * np.pi) * coeff['sigma_2'] * coeff['amplitude_2']
    deltaEW2 = np.sqrt((perr[2] / coeff['sigma_2'])**2 + (perr[0] / coeff['amplitude_2'])**2 - (2 * perr[1] * perr[3])**2)
    z2 = coeff['x0_1'] / rest_wavelength_2 - 1

    return EW1 / (1 + z1), deltaEW1 / (1 + z1), EW2 / (1 + z2), deltaEW2 / (1 + z2)


